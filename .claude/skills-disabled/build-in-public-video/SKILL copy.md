---
name: build-in-public-video
description: Generate a build in public video based on ./video.topic.yaml. Reads the topic, then produces a video file (e.g. MP4) with remotion. Trigger when the user asks to generate a video.
---

# build-in-public-video
- Generate a video based on `./video.topic.yaml` 
- Add a voice-over narration track.
- make it 1-2 minutes long (may be overridden by the user)

## How to Generate Voice-over Narration
Generate a voice-over audio track for the video, split per scene. Use the macOS `say` command (so it can be customized via system speech settings), then convert each clip to MP3 with ffmpeg.

Steps:
1. Write one text file per scene under `public/scenes/` (e.g. `scene1.txt` … `sceneN.txt`). Use short, spoken-friendly sentences. Replace symbols and underscores with spoken forms (e.g. write "a11y parse" instead of "a11y_parse").

2. For each scene file, generate the audio:
   say -f public/scenes/sceneN.txt -o public/scenes/sceneN.aiff
   ffmpeg -y -i public/scenes/sceneN.aiff public/scenes/sceneN.mp3

3. Read each scene's duration from ffmpeg's output (the `Duration:` line) and use those values to size each `<Sequence>` in the Remotion composition. Layer each `public/scenes/sceneN.mp3` as an `<Audio>` (from `@remotion/media`) inside its corresponding `<Sequence>`, so each scene plays its own narration.

Output: `public/scenes/sceneN.mp3` files, one per scene.


## Slides Generation into PDF 
After the Remotion video is generated, create still slides at the end of each scene and assemble them into a PDF.

1. Read `src/Composition.tsx` to extract:
   - Each scene's duration in frames (e.g. SCENE1_FRAMES, SCENE2_FRAMES, …)
   - Each scene's offset in the composition (cumulative sum of prior durations)
   - The fade-out duration used by `fadeInOut(frame, durationInFrames, fadeFrames)` — typically 18 frames
   - The composition id and entry file (e.g. `MyComp` in `src/index.ts`)

2. For each scene N, compute the capture frame as the last fully-opaque frame BEFORE the fade-out begins:
   - `captureFrameN = sceneOffsetN + sceneDurationN - fadeFrames - 1`
   - This ensures the slide shows the scene fully rendered, with all in-scene animations settled and no fade transparency.

3. Render one PNG per scene with Remotion's still command:
    - `npx remotion still <entry> <comp-id> out/sceneN.png --frame=<captureFrameN> --image-format=png`
    - Run sequentially (or chained with &&) so the bundle is reused. 
    - Output goes to `out/scene1.png` … `out/sceneN.png` at the composition's native resolution (e.g. 1920×1080).

4. Combine the PNGs into a single multi-page PDF using ImageMagick, in scene order:
    - `magick out/scene1.png out/scene2.png … out/sceneN.png out/slides.pdf`

5. Verify `out/slides.pdf` exists and report the per-scene capture frames plus the PDF path back to the user.




# misc Remarks 
- if you need ffmpeg, use https://www.npmjs.com/package/ffmpeg-static
- voice over will be generated using macosx 'say' command, so you can customize the voice and speed by changing the system settings for 'say'
- to render the video, use remotion's render command, e.g. `npx remotion render src/Video.tsx out/video.mp4`
