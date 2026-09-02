# PaveFOD dataset card

## Summary

PaveFOD is a controlled outdoor dataset for pixel-level and object-level
foreign object debris detection on pavement. It contains fixed-oblique RGB
views with pavement markings, joints, wear, shadows, and heterogeneous surface
texture.

| Property | Value |
| --- | ---: |
| RGB images | 87 |
| Image resolution | 2560 x 1440 |
| Object-free training images | 4 |
| Anomalous validation images | 41 |
| Anomalous test images | 42 |
| Validation instances | 95 |
| Test instances | 103 |
| Object types | 8 |
| Nominal acquisition ranges | 5-30 m |

The eight canonical object labels are `cylinder`, `fuelcap`, `golf`,
`lampscrew`, `screw`, `socket`, `strip`, and `wrench`. The mixed-range group is
represented by distance code `30` and nominal range `5-30m`.

## Release layout

```text
PaveFOD-v1.0.0/
  05|10|15|20|25|30/
    train|val|test/
      good|anomaly/
        rgb/
        gt/
        instance_gt/
  roi/
    ground_roi_hard.png
    ground_roi_soft.png
  metadata.csv
  instances.csv
  dataset_manifest.json
  SHA256SUMS
  README.md
  LICENSE.md
```

RGB images, binary masks, and instance masks use the same sample stem and PNG
extension. Instance-mask values are zero for background and positive integers
for individual objects. `instances.csv` maps those integer IDs to canonical
object labels, mask areas, bounding boxes, and measured depths.

## Privacy and provenance

The released images are privacy-filtered derivatives of the research frames.
Pixels inside the verified pavement ROI are unchanged. Pixels outside the ROI
are set to white because the original background contains people and unrelated
public-space activity. Original camera IP addresses and capture timestamps are
removed from filenames and metadata. This processing retains every evaluation
mask and all pixels within the evaluation region.

PaveFOD was collected by the project team in a controlled outdoor paved scene.
The release contains only project-owned PaveFOD content. VisDrone and UAVVaste
are not included.

## Intended use

The dataset supports research on open-set FOD detection, anomaly localization,
small-object discovery, and fixed-camera pavement inspection. It is not a
certification dataset and does not establish compliance with any aviation
equipment standard.

## Limitations

PaveFOD is small and scene-specific. Its fixed viewpoint, controlled object
placement, geographic setting, surface appearance, weather, and illumination
do not cover the full range of airport operations. Results should not be
interpreted as deployment readiness. Models should be validated on additional
sites, cameras, conditions, and operational hazards before practical use.

## License

PaveFOD v1.0.0 is released under CC BY-NC 4.0. See `LICENSE.md` in the archive
or [`DATASET_LICENSE.md`](../DATASET_LICENSE.md).

