# Figures for Report and Presentation

This folder contains all plots, qualitative outputs, and visual materials used in the report or presentation slides. **Include only final, report-grade assets here.**

## Expected Contents

### Main Metrics Plots

- **main_metrics.png** — Bar or line chart showing the headline metrics from `main_results.csv`
    - Suggested layout: FPS, latency, spatial accuracy, distance accuracy, safest-side accuracy, hazard mention rate, VLM usefulness
    - Should directly match numbers in the report

### Ablation Study Plots

- **ablation_study.png** — Comparative bar chart showing impact of each ablation from `ablations.csv`
    - X-axis: variant names (no_depth, no_temporal, no_road, no_vlm, etc.)
    - Y-axis: metric values
    - Overlay full model value as a baseline
    - Should highlight which components matter most

- **ablation_flicker_rate.png** — Time-series or bar chart of instruction flicker rate across ablations
    - Highlights temporal reasoning impact

- **ablation_fps.png** — FPS comparison across ablations
    - Shows which components add latency

### Qualitative Image Results

- **qualitative_image_001.png** — Annotated image output from test image 001
    - Should show: original image, bounding boxes, direction/distance labels, final caption overlay
    - Filename format: `qualitative_image_NNN.png` where NNN matches `image_id` from annotations

- **qualitative_image_002.png** — Similar for test image 002

- ... (one per test image if possible, or at least 2-3 representative examples)

### Qualitative Video Results

- **qualitative_video_sample1_frame_0010.png** — Frame from video run at specific timepoint
    - Should show: detections, road overlay, instruction caption at that frame
    - Good to include a few frames showing progression of instruction

- **qualitative_video_sample1_frame_0025.png** — Another frame from same video
    - Can show transition between instructions or handling of multiple hazards

- Filename format: `qualitative_video_<run>_frame_<NNNN>.png`

### Performance/Latency Plots

- **latency_breakdown.png** — Stacked bar chart showing:
    - Detection latency
    - Depth latency
    - Spatial reasoning latency
    - Planning latency
    - Caption generation latency
    - Total

- **fps_over_video.png** — (Optional) Line plot of FPS sampled over a long video run
    - Shows if performance degrades with time

### Architecture or Process Diagrams

- **pipeline_architecture.png** — Block diagram of the full system
    - Can be copied from README or recreated as a clean figure

- **rule_based_planner_flowchart.png** — (Optional) Flowchart of the 8-layer decision rules

### Tables as Figures

If you use tables in the report, include them:

- **results_table.png** — Screenshot or exported image of main results table from report
- **ablation_table.png** — Screenshot of ablation results table
- **model_specifications.png** — Screenshot of model specs table (from README)

## File Naming Conventions

- Use lowercase, underscores, and clear descriptive names
- Include metric or test name: `spatial_accuracy`, `flicker_rate`, `qualitative_image`
- Date/version optional but helpful: `main_metrics_v2.png`
- Never use spaces; use underscores
- All PNG (preferred for crisp vector plots) or JPG (for photographic/screenshot content)

## Generation Workflow

1. **For metric plots**: Export from Python (matplotlib, plotly) after running evaluation
2. **For qualitative outputs**: Copy or screenshot from `05_results/figures/outputs/` after running the pipeline
3. **For tables**: Export CSV to PNG using Python or office tools, or screenshot from report
4. **For diagrams**: Use draw.io, Excalidraw, or Mermaid and export as PNG

## Quality Standards

- Plots should have clear labels, legends, and units
- Font size should be readable at report/slide viewing distance
- Use a consistent color scheme across related plots
- Include data source or test set size in plot title or caption
- Ensure all axis scales are appropriate (don't truncate zero if it would mislead)

## Integration with Report

- Each figure should be referenced in the report text
- Caption should explain what is shown and key takeaway
- Ablation figures should directly support claims about component importance
- Qualitative figures should show diversity of scene types and success/failure modes

---

See parent folder `../logs/` for evaluation details that produced these numbers.
