import os, json, base64, requests, streamlit as st

REPO_API = "https://api.github.com/repos/Darkjay123/story-studio"
def _get_token():
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        try:
            tok = st.secrets.get("GITHUB_TOKEN", "")
        except Exception:
            tok = ""
    return tok

TOKEN = _get_token()
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}

st.set_page_config(page_title="Story Studio", page_icon="🎬", layout="centered")
st.title("🎬 Story Studio")
st.caption("Upload a keyframe, describe the motion, queue the render. Clips land in the repo when done.")

def get_queue():
    r = requests.get(f"{REPO_API}/contents/jobs.json", headers=HEADERS)
    r.raise_for_status()
    return json.loads(base64.b64decode(r.json()["content"]))["jobs"], r.json()["sha"]

def put_jobs(jobs, sha, msg):
    r = requests.put(f"{REPO_API}/contents/jobs.json", headers=HEADERS, json={
        "message": msg, "sha": sha,
        "content": base64.b64encode(json.dumps({"jobs": jobs}, indent=2).encode()).decode(),
    })
    r.raise_for_status()

with st.form("submit", clear_on_submit=True):
    img = st.file_uploader("Reference keyframe (png/jpg)", type=["png", "jpg", "jpeg"])
    prompt = st.text_area("Motion prompt", height=90,
        placeholder="The girl steps forward away from the yellow danfo bus and walks confidently through a Lagos market street, her braids swaying, warm golden light, cinematic 2D anime animation")
    engine_label = st.radio("Model",
        ["Wan 2.2 (hero: best character fidelity, slower)",
         "LTX-Video 2B (bulk: fast, 30fps, watch for character drift)"],
        horizontal=True)
    res = st.selectbox("Resolution",
        ["480x480 (fast test)", "704x1280 (production portrait)"])
    c1, c2, c3, c4 = st.columns(4)
    seconds = c1.number_input("Seconds", 2.0, 10.0, 5.0, 0.5)
    fps = c2.selectbox("FPS", [16, 24], index=1)
    steps = c3.number_input("Steps", 10, 50, 30)
    seed = c4.number_input("Seed", 0, 999999, 7)
    submit = st.form_submit_button("Queue render", use_container_width=True, type="primary")

if submit:
    if not (img and prompt):
        st.error("Need a keyframe and a prompt.")
    else:
        jobs, sha = get_queue()
        jid = f"job-{len([j for j in jobs if j['id'].startswith('job-2026')])+1:04d}-{img.name.split('.')[0][:16]}"
        img_path = f"uploads/{jid}.png"
        r = requests.put(f"{REPO_API}/contents/{img_path}", headers=HEADERS, json={
            "message": f"keyframe {jid}",
            "content": base64.b64encode(img.getvalue()).decode(),
        })
        r.raise_for_status()
        is_ltx = engine_label.startswith("LTX")
        width, height = (704, 1280) if "704" in res else (480, 480)
        jobs.append({"id": jid, "status": "queued", "model": "ltx_bulk" if is_ltx else "wan_hero",
                     "engine": "ltx" if is_ltx else "wan",
                     "keyframe": img_path, "prompt": prompt,
                     "negative": "blurry, low quality, deformed face, extra limbs, text, watermark",
                     "width": width, "height": height, "seconds": float(seconds), "fps": int(fps),
                     "steps": int(steps), "guidance": 3.0 if is_ltx else 5.0, "seed": int(seed),
                     "created": "2026-10-17", "finished": None, "output": None})
        put_jobs(jobs, sha, f"studio: queue {jid}")
        st.success(f"Queued {jid}. The worker picks it up on its next Kaggle run.")
        st.rerun()

st.subheader("Queue")
jobs, _ = get_queue()
if not jobs:
    st.info("No jobs yet.")
for j in reversed(jobs):
    icon = {"queued": "⏳", "running": "🎥", "done": "✅", "failed": "❌"}.get(j["status"], "?")
    tag = "LTX" if j.get("engine") == "ltx" else "Wan"
    line = f"{icon} **{j['id']}** · {tag} — {j['status']}"
    if j.get("output"):
        line += f" · [{j['output']}](https://github.com/Darkjay123/story-studio/raw/main/{j['output']})"
    if j.get("error"):
        line += f" · {j['error'][:120]}"
    st.markdown(line)
