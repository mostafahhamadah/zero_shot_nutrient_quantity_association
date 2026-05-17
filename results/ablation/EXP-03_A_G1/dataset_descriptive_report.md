# Dataset-level descriptive report — EXP-03

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 903
- True positives: 215
- False positives: 688
- False negatives: 651
- Dataset-level **precision = 0.2381**, **recall = 0.2483**, **F1 = 0.2431**
- Mean per-image F1: 0.3347
- Median per-image F1: 0.3300
- Standard deviation of per-image F1: 0.2770

## Performance characterisation

The experiment is limited primarily by precision and recall in roughly balanced proportion. The 0.28 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id   f1  tp  fp  fn
       1.png 1.00   4   1   1
       6.png 1.00   6   0   0
    109.jpeg 0.89   8   4   1
     101.png 0.82   9   2   2
      17.png 0.75   3   4   1

## Worst-performing images

    image_id  f1  tp  fp  fn
     106.png 0.0   0   8   2
    111.jpeg 0.0   0   2   1
    112.jpeg 0.0   0   2   1
    113.jpeg 0.0   0  22  20
    118.jpeg 0.0   0  14  20
