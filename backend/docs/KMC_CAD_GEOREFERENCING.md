# Georeferencing the KMC AutoCAD ward sheets (wards 36, 61, 67, 104)

These four sheets have no embedded coordinates. The pipes have already been extracted from the PDF vectors. All that's missing is where the sheet sits on the ground. You supply that as 6–8 ground control points (GCPs) per sheet in QGIS. Expect about 15 minutes per ward.

Wards 107 and 108 are GeoPDFs and need nothing from you.

## What you need

- QGIS 3.34 LTR or newer
- The reference images, already generated: `backend/data/processed/kmc_extracted/cad/render/ward036.png`, `ward061.png`, `ward067.png`, `ward104.png`

Don't use the original PDFs or a PNG you export yourself. The script relies on the exact pixel↔PDF mapping of these renders, and has verified it for each sheet: the ward boundary lands on the drawn magenta boundary for 99.7–100% of points.

## Steps (repeat per ward)

1. **Basemap.** In the QGIS Browser panel, expand *XYZ Tiles* and double-click *OpenStreetMap*. Zoom to the ward.
2. **Open the Georeferencer.** *Layer ▸ Georeferencer…*. Click *Open Raster* and pick `render/wardNNN.png`. If asked for a CRS, cancel or accept anything: the image is treated as plain pixels.
3. **Transformation settings** (gear icon):
   - Transformation type: **Helmert**. This allows scale, rotation and shift only, which is the right model for a CAD drawing.
   - Target CRS: EPSG:4326 or EPSG:3857, either works.
   - Leave the output raster empty. You don't need to run the georeferencing itself.
4. **Add points.** Press *Add Point*, then click a feature in the image:
   - A *From Map Canvas* dialog appears. Click the same feature on the OSM map, then press OK.
   - Use road-junction centres. The CAD sheets draw both road edges, and OSM draws the centreline, so click the middle of the junction in both.
   - Pick 6–8 junctions spread over the whole ward, including its far corners. Don't put them all along one road (nearly in a line).
   - Pick junctions of named roads you can identify in both views. Named roads on each sheet:
     - **36 (Sealdah):** Beliaghata Road, Canal East Road, Beliaghata Bridge, Maharani Swarnamoyee Road, the Circular Canal crossings
     - **61 (Park Street):** Park Street, Elliot Road, A.J.C. Bose Road, Rafi Ahmed Kidwai Road, Park Lane, Royd Street
     - **67 (Kasba):** Bose Pukur Road, Rash Behari Connector, Picnic Garden Road, Swinhoe Lane, the railway line on the west edge
     - **104 (Kalikapur):** Garfa 1st–4th Lanes, Naskar Para Road, Vivekananda Sarani, Kalikapur
5. **Check residuals** in the GCP table (units are pixels; 1 px ≈ 0.35–0.5 m on these renders):
   - Under about 10 px everywhere is good.
   - A single point far above the others is almost always a mis-click. Delete it and re-pick.
6. **Save.** Choose *File ▸ Save GCP Points As…* and save to `backend/data/processed/kmc_extracted/cad/gcp/wardNNN.points`, for example `ward036.points`. The number must match the image name.

## Apply

```sh
cd backend
../.venv/bin/python scripts/extract_kmc_cad.py apply
```

For each sheet the script fits the transform and **rejects** it, writing nothing, if any of these checks fail:

| Check | Limit |
|---|---|
| GCP count | at least 4 |
| RMS residual | at most 5 m |
| Worst leave-one-out error (catches one bad point hidden by the others) | at most 10 m |
| Scale vs. the sheet's printed ward area | within 10% |
| Rotation vs. the sheet's north arrow | within 10° |

It also reports the distance from the pipes to OSM road centrelines, as an independent sanity check. Accepted sheets are merged into `data/gis/drainage_network.geojson`, alongside wards 107 and 108. The full report is in `cad/apply_report.json`.

If a sheet is rejected, the report says which check failed. Fix or replace GCPs in QGIS, re-save, and run `apply` again.

## What was extracted per sheet (review before trusting)

The QA overlays are in `cad/qa/wardNNN_qa.png`. Each pipe is coloured by its assigned diameter; black means no diameter was found, and none is guessed.

| Ward | Pipes | With diameter | Diameter source | Coverage caveat |
|---|---|---|---|---|
| 104 | 303 segments | 94% of length | Map labels, e.g. "(300mm Ø)" | none known |
| 61 | 112 segments | 56% of length | Layer name (checked against labels: 42 of 43 agree) | Trunk/brick sewers on the road layer have no mm size |
| 36 | 89 segments | 54% of length | Map labels | The `Z DRAIN` layer (green dashed) is kept as `conduit_type = drain` with no diameter. Its exact meaning isn't stated on the sheet |
| 67 | 29 segments | 0% | none | **Trunk sewers only.** The branch network is drawn in the same style as road edges, sized in inches, and is not extracted. Treat ward 67 as partial coverage |
