# Dataset-level descriptive report — EXP-04

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 912
- True positives: 362
- False positives: 550
- False negatives: 504
- Dataset-level **precision = 0.3969**, **recall = 0.4180**, **F1 = 0.4072**
- Mean per-image F1: 0.5012
- Median per-image F1: 0.5700
- Standard deviation of per-image F1: 0.3425

## Performance characterisation

The experiment is limited primarily by precision and recall in roughly balanced proportion. The 0.34 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id  f1  tp  fp  fn
       1.png 1.0   4   1   1
    109.jpeg 1.0   9   3   0
    110.jpeg 1.0   6   0   0
      15.png 1.0   9   0  33
       6.png 1.0   6   0   0

## Worst-performing images

    image_id  f1  tp  fp  fn
     106.png 0.0   0   8   2
    111.jpeg 0.0   0   2   1
    112.jpeg 0.0   0   2   1
    118.jpeg 0.0   0  15  20
    119.jpeg 0.0   0  12  20
