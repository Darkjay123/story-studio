# story-studio
Web studio + batch queue for AI story-video generation (Wan 2.2 TI2V-5B hero shots, LTX-Video 2B bulk shots). Free stack: Hugging Face Space (Streamlit frontend) -> jobs.json queue -> Kaggle notebook worker (free T4 GPU).

Flow: submit jobs in the studio (reference image + prompt) -> they land in jobs.json -> the Kaggle worker cell pulls every queued job, renders, saves mp4s under outputs/, and marks jobs done. Queue everything at night, collect clips in the morning.

## Job schema
id, status (queued|running|done|failed), model (wan_hero|ltx_bulk), keyframe (repo path), prompt, negative, width, height, seconds, fps, steps, guidance, seed, created, finished, output.

## Setup
- Kaggle notebook secret: GITHUB_TOKEN (fine-grained, Contents read/write on this repo).
- HF Space secret: GITHUB_TOKEN (same purpose, for submitting jobs).
