# Documentation Assets

This directory houses visual assets and documentation screenshots for the repository.

## Adding the UI Screenshot

To update or add the UI screenshot for `README.md`:
1. Start the Streamlit frontend locally:
   ```bash
   streamlit run src/ui/app.py
   ```
2. Enter a statutory query, for example:
   > *"Does public legislation require or imply expectations related to risk identification or monitoring in policing, and how are these reflected in oversight activities?"*
3. Once the CRAG pipeline completes generation, capture a cropped screenshot focused on the response card displaying:
   - The 🟢 High Confidence indicator badge
   - The statutory citations (e.g. `[CSPA s.102(4)(a)]`)
   - The Sources section
4. Optimize the image (PNG format, keep under 500 KB) and save it as:
   ```
   docs/ui_preview.png
   ```
