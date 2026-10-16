# ===== 3) story-studio batch worker: pull queue, render, push results =====
# One-time setup in Kaggle: Add-ons -> Secrets -> add GITHUB_TOKEN (fine-grained PAT
# with Contents read/write on Darkjay123/story-studio).

import json, os, subprocess, sys, datetime

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
    out_path = os.path.join(WORK, j.get("output", ""))
    if j["status"] == "running" and not (j.get("output") and os.path.exists(out_path)):
        j["status"] = "queued"

def save_queue(commit_msg):
    with open(JOBS, "w") as f:
        json.dump({"jobs": queue}, f, indent=2)
    sh(f'git add {JOBS} && git commit -m "{commit_msg}"')
    sh("git push origin main")

save_queue(f"worker: resume stale jobs {datetime.datetime.utcnow().isoformat()[:19]}")

pending = [j for j in queue if j["status"] == "queued"]
print(f"pending jobs: {len(pending)}")

from diffusers import WanPipeline
from diffusers.utils import load_image, export_to_video
import torch

pipe = None
def get_pipe():
    global pipe
    if pipe is None:
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
    return pipe

for j in pending:
    try:
        j["status"] = "running"
        save_queue(f"worker: start {j['id']}")
        img_path = os.path.join(WORK, j["keyframe"])
        image = load_image(img_path).resize((j["width"], j["height"]))
        frames = get_pipe()(
            prompt=j["prompt"], negative_prompt=j["negative"], image=image,
            height=j["height"], width=j["width"],
            num_frames=int(j["seconds"] * j["fps"]), fps=j["fps"],
            guidance_scale=j["guidance"], num_inference_steps=j["steps"],
            generator=torch.Generator("cuda").manual_seed(j["seed"]),
        ).frames[0]
        out_rel = f"outputs/{j['id']}.mp4"
        export_to_video(frames, out_rel, fps=j["fps"])
        j["output"] = out_rel
        j["status"] = "done"
        j["finished"] = datetime.datetime.utcnow().isoformat()[:19]
        sh(f'git add "{out_rel}" && git commit -m "worker: output {j[\'id\']}"')
        save_queue(f"worker: done {j['id']}")
        print(f"done {j['id']}")
    except Exception as e:
        j["status"] = "failed"
        j["error"] = str(e)[:300]
        j["finished"] = datetime.datetime.utcnow().isoformat()[:19]
        save_queue(f"worker: failed {j['id']}")
        print(f"FAILED {j['id']}: {e}", file=sys.stderr)

print("worker run complete")
