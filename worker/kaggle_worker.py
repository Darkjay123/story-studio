# ===== 3) story-studio batch worker: pull queue, render, push results =====
# One-time setup in Kaggle: Add-ons -> Secrets -> add GITHUB_TOKEN (fine-grained PAT
# with Contents read/write on Darkjay123/story-studio).

import traceback
import json, os, subprocess, sys, datetime, time, math

import os
GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    from kaggle_secrets import UserSecretsClient
    GH_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
REPO = "Darkjay123/story-studio"

WORK = "/kaggle/working/story-studio"
JOBS = os.path.join(WORK, "jobs.json")

def sh(cmd, **kw):
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)

# fresh clone every run
sh(f"rm -rf {WORK}")
sh(f"git clone https://{GH_TOKEN}@github.com/{REPO}.git {WORK}")
os.chdir(WORK)
sh('git config user.email "worker@kaggle.local" && git config user.name "kaggle-worker"')
sh("git pull --rebase origin main")

with open(JOBS) as f:
    queue = json.load(f)["jobs"]

# recovery: running jobs with no saved output go back to queued (session died last run)
for j in queue:
    out_path = os.path.join(WORK, j.get("output") or "")
    if j["status"] == "running" and not (j.get("output") and os.path.exists(out_path)):
        j["status"] = "queued"

def save_queue(commit_msg):
    with open(JOBS, "w") as f:
        json.dump({"jobs": queue}, f, indent=2)
    # no-op when nothing changed (fresh clone of an unchanged queue); plain commit exits 1 there
    sh('git add ' + JOBS + ' && git diff --cached --quiet || git commit -m "' + commit_msg + '"')
    # 17 Oct: a push racing newer commits on main used to kill the whole worker
    # mid-loop and leave jobs.json stuck on "running" forever. Rebase onto the
    # remote, retry, and log instead of raising so status always gets a chance to land.
    for attempt in range(3):
        r = subprocess.run("git pull --rebase --autostash origin main && git push origin main",
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            return
        print(f"PUSH-RETRY {attempt + 1}: {(r.stdout or '')[-800:]}", flush=True)
        time.sleep(15)
    r = subprocess.run("git pull --rebase --autostash origin main && git push origin main",
                       shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print("PUSH-FAILED: " + (r.stdout or "")[-800:], file=sys.stderr, flush=True)

save_queue(f"worker: resume stale jobs {datetime.datetime.utcnow().isoformat()[:19]}")

pending = [j for j in queue if j["status"] == "queued"]
print(f"pending jobs: {len(pending)}")

# ftfy fix (16 Oct): the Wan tokenizer path references ftfy and the Kaggle image
# lacks it, which killed Version 4 with "name 'ftfy' is not defined" at 12:29 UTC.
# Install it and expose it as a builtin so any bare ftfy reference resolves.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy"], check=True)
import ftfy, builtins
builtins.ftfy = ftfy

# ===== round-8 VAE diagnostics (16 Oct): V6 died mid-decode with
# "tensor a (2) must match tensor b (4) at dim 2" inside WanResidualUpBlock's
# avg_shortcut add. The same code passes a CPU repro, so instrument the real
# loaded model: log the VAE config, per-chunk up-block shapes and cache-slot
# states, and guard the add so a mismatch cannot kill the job.
import diffusers.models.autoencoders.autoencoder_kl_wan as _wan_vae

_diag = {"chunk": 0}

def _slot_state(slot):
    if slot is None:
        return "None"
    if isinstance(slot, str):
        return "Rep"
    return "T%d" % slot.shape[2]

_orig_decode = _wan_vae.AutoencoderKLWan.decode

def _diag_decode(self, z, return_dict=True):
    if _diag["chunk"] == 0:
        c = self.config
        print("[VAE-DIAG] decode in", tuple(z.shape), flush=True)
        print("[VAE-DIAG] cfg dim_mult=%s decoder_base_dim=%s temperal_downsample=%s "
              "temperal_upsample=%s is_residual=%s num_res_blocks=%s" % (
                  c.dim_mult, c.decoder_base_dim, c.temperal_downsample,
                  self.temperal_upsample, c.is_residual, c.num_res_blocks), flush=True)
        for bi, ub in enumerate(self.decoder.up_blocks):
            print("[VAE-DIAG] up_block %d: %s in=%s out=%s upsampler=%s dup=%s" % (
                bi, type(ub).__name__, ub.in_dim, ub.out_dim,
                getattr(getattr(ub, "upsampler", None), "mode", None),
                getattr(getattr(ub, "avg_shortcut", None), "factor", None)), flush=True)
    _diag["chunk"] += 1
    return _orig_decode(self, z, return_dict=return_dict)

_wan_vae.AutoencoderKLWan.decode = _diag_decode

_orig_upblock = _wan_vae.WanResidualUpBlock.forward

def _diag_upblock(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    x_copy = x.clone()
    for resnet in self.resnets:
        x = resnet(x, feat_cache, feat_idx) if feat_cache is not None else resnet(x)
    ups_state = None
    if self.upsampler is not None:
        ups_state = _slot_state(feat_cache[feat_idx[0]]) if feat_cache is not None else "nocache"
        x = self.upsampler(x, feat_cache, feat_idx) if feat_cache is not None else self.upsampler(x)
    if self.avg_shortcut is not None:
        sc = self.avg_shortcut(x_copy, first_chunk=first_chunk)
        if _diag["chunk"] <= 3:
            print("[VAE-DIAG] upblock chunk=%d first=%s inT=%d mainT=%d scT=%d ups_slot=%s" % (
                _diag["chunk"], first_chunk, x_copy.shape[2], x.shape[2], sc.shape[2], ups_state), flush=True)
        if sc.shape[2] != x.shape[2]:
            print("[VAE-GUARD] shortcut/main mismatch mainT=%d scT=%d chunk=%d ups_slot=%s "
                  "- realigning instead of crashing" % (
                      x.shape[2], sc.shape[2], _diag["chunk"], ups_state), flush=True)
            if sc.shape[2] > x.shape[2] and sc.shape[2] % x.shape[2] == 0:
                x = x.repeat_interleave(sc.shape[2] // x.shape[2], dim=2)
            elif x.shape[2] > sc.shape[2] and x.shape[2] % sc.shape[2] == 0:
                sc = sc.repeat_interleave(x.shape[2] // sc.shape[2], dim=2)
        x = x + sc
    return x

_wan_vae.WanResidualUpBlock.forward = _diag_upblock
print("[VAE-DIAG] instrumentation active", flush=True)

from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from huggingface_hub import snapshot_download
import torch

pipe = None
def get_pipe():
    global pipe
    if pipe is None:
        # 5B TI2V image jobs must use WanImageToVideoPipeline (diffusers issue #13258);
        # WanPipeline rejects image= and crashes. Proven load recipe from the test session.
        local = snapshot_download("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        # 16 Oct round-11: at 720x1280 the balanced 2-GPU split OOMs on the FIRST
        # transformer forward (GPU 1: 14.39 GiB used of 14.56, tried 1.96 GiB more;
        # killed test-0002, Version 9). The 720p footprint no longer fits a 15 GB
        # T4 half alongside the text encoder, so execute layer-by-layer with CPU
        # offload instead: peak GPU memory is one block's activations, which always
        # fits. Slower per step, but it is the standard diffusers recipe for this.
        pipe = WanImageToVideoPipeline.from_pretrained(local, torch_dtype=torch.float16)
        pipe.enable_sequential_cpu_offload()
        # NOTE: do NOT enable_tiling() on the Wan 2.2 patchify VAE. Its tiled
        # encode/decode path feeds raw 3-channel tiles straight into conv_in and
        # skips the patchify step (patch_size=2 -> 12 channels), which crashes with
        # "expected input[1, 3, 3, 258, 258] to have 12 channels" (killed Version 5).
        # The causal VAE streams frames anyway, so tiling is unnecessary at 480p.
        print("I2V-LOADED")
    return pipe

ltx_pipe = None
def get_ltx_pipe():
    global ltx_pipe
    if ltx_pipe is None:
        from diffusers import LTXImageToVideoPipeline
        # LTX-Video 2B: the fast bulk-shot engine. fp16 because T4 has no bf16.
        # 18 Oct root-cause: V12/V13 both died ~8 min in with "Read-only file
        # system" from the download layer. The 0.9.6-dev repo is a ~48 GB fp32
        # class checkout (same shape as 0.9.7-dev: 30 GB transformer + 20 GB
        # T5-XXL). snapshot_download pulls ALL of it, fills the container's
        # overlay disk mid-download, and the FS flips read-only (the kernel
        # teardown chmod hit the same wall). Wan works only because its repo is
        # 34 GB and still fits. 0.9.5 is diffusers-format and 25.6 GB total
        # (3.85 GB 2B transformer + 2.49 GB VAE + 19 GB T5 fp32), smaller than
        # the Wan download that already succeeds, so use it and log disk headroom.
        import shutil
        free = shutil.disk_usage("/").free / 1e9
        print(f"LTX-DISK free={free:.1f}GB", flush=True)
        local = snapshot_download("Lightricks/LTX-Video-0.9.5")
        ltx_pipe = LTXImageToVideoPipeline.from_pretrained(local, torch_dtype=torch.float16)
        ltx_pipe.enable_sequential_cpu_offload()
        # LTX's VAE is a normal conv VAE (not Wan's patchify one), so tiling is safe
        # here and keeps 704x1280 decode memory sane.
        ltx_pipe.vae.enable_tiling()
        print("LTX-LOADED", flush=True)
    return ltx_pipe

for j in pending:
    try:
        j["status"] = "running"
        save_queue(f"worker: start {j['id']}")
        img_path = os.path.join(WORK, j["keyframe"])
        if j.get("engine", "wan") == "ltx":
            # LTX wants width/height divisible by 32 and (num_frames - 1) % 8 == 0.
            w = (int(j["width"]) // 32) * 32
            h = (int(j["height"]) // 32) * 32
            image = load_image(img_path).resize((w, h))
            # 18 Oct: a 30s @ 30fps job (897 frames) OOM'd on T4 (tried 4.55 GiB, 3.52
            # free) because LTX denoises every frame in one pass. Long jobs now render
            # in <=8s segments (241 frames, the size already proven to fit) and each
            # later segment is conditioned on the previous segment's last frame, so
            # motion carries across the seam; its duplicated first frame is dropped.
            fps_j = int(j["fps"])
            total = float(j["seconds"])
            if total <= 8.0:
                segs = [total]
            else:
                n = int(math.ceil(total / 8.0 - 1e-6))
                segs = [total / n] * n
            parts = []
            cond = image
            for si, seg in enumerate(segs):
                nf = int(seg * fps_j); nf = ((nf - 1) // 8) * 8 + 1
                print(f"ltx segment {si + 1}/{len(segs)} frames={nf}", flush=True)
                out = get_ltx_pipe()(
                    image=cond, prompt=j["prompt"], negative_prompt=j["negative"],
                    height=h, width=w, num_frames=nf, frame_rate=fps_j,
                    num_inference_steps=j["steps"], guidance_scale=float(j.get("guidance", 3.0)),
                    # decode_timestep/noise_scale: the 0.9.6-dev recommended decode recipe.
                    decode_timestep=0.05, decode_noise_scale=0.025,
                    generator=torch.Generator("cuda").manual_seed(j["seed"]),
                ).frames[0]
                parts.append(out if si == 0 else out[1:])
                if si + 1 < len(segs):
                    cond = out[-1]
                    torch.cuda.empty_cache()
            # 19 Oct fix: diffusers may return a segment as a list of PIL images
            # instead of a tensor (LTX 0.9.5 did), so torch.cat died with
            # "expected Tensor as element 0 in argument 0, but got list" after
            # every segment had already rendered. Lists concatenate as lists;
            # tensors keep torch.cat.
            if len(parts) == 1:
                frames = parts[0]
            elif isinstance(parts[0], (list, tuple)):
                frames = [f for part in parts for f in part]
            else:
                frames = torch.cat(parts, dim=0)
        else:
            image = load_image(img_path).resize((j["width"], j["height"]))
            # Wan needs 4n+1 frames; snap (80 -> 81). fps belongs to export_to_video, not the pipe call.
            # 18 Oct: same segmenting as the LTX fix. A 30s @ 24fps job is 721 frames in
            # one pass, a known OOM risk on T4 x2; 8s segments (193 frames) already fit.
            total = float(j["seconds"])
            if total <= 8.0:
                segs = [total]
            else:
                n = int(math.ceil(total / 8.0 - 1e-6))
                segs = [total / n] * n
            parts = []
            cond = image
            for si, seg in enumerate(segs):
                nf = int(seg * j["fps"]); nf = (nf // 4) * 4 + 1
                print(f"wan segment {si + 1}/{len(segs)} frames={nf}", flush=True)
                out = get_pipe()(
                    prompt=j["prompt"], negative_prompt=j["negative"], image=cond,
                    height=j["height"], width=j["width"], num_frames=nf,
                    guidance_scale=j["guidance"], num_inference_steps=j["steps"],
                    generator=torch.Generator("cuda").manual_seed(j["seed"]),
                ).frames[0]
                parts.append(out if si == 0 else out[1:])
                if si + 1 < len(segs):
                    cond = out[-1]
                    torch.cuda.empty_cache()
            # 19 Oct fix: diffusers may return a segment as a list of PIL images
            # instead of a tensor (LTX 0.9.5 did), so torch.cat died with
            # "expected Tensor as element 0 in argument 0, but got list" after
            # every segment had already rendered. Lists concatenate as lists;
            # tensors keep torch.cat.
            if len(parts) == 1:
                frames = parts[0]
            elif isinstance(parts[0], (list, tuple)):
                frames = [f for part in parts for f in part]
            else:
                frames = torch.cat(parts, dim=0)
        out_rel = f"outputs/{j['id']}.mp4"
        # 16 Oct round-9: a fresh clone has no outputs/ (gitignored), so export_to_video
        # died with "[Errno 2] The directory does not exist" (killed Version 7).
        os.makedirs(os.path.dirname(out_rel), exist_ok=True)
        export_to_video(frames, out_rel, fps=j["fps"])
        j["output"] = out_rel
        j["status"] = "done"
        j["finished"] = datetime.datetime.utcnow().isoformat()[:19]
        msg = "worker: output " + j["id"]
        sh(f'git add "{out_rel}" && git commit -m "{msg}"')
        save_queue(f"worker: done {j['id']}")
        print(f"done {j['id']}")
    except Exception as e:
        j["status"] = "failed"
        traceback.print_exc()
        j["error"] = str(e)[:300]
        j["finished"] = datetime.datetime.utcnow().isoformat()[:19]
        save_queue(f"worker: failed {j['id']}")
        print(f"FAILED {j['id']}: {e}", file=sys.stderr)

print("worker run complete")
