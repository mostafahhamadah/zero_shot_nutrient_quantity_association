# Dataset-level descriptive report — EXP-05

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 628
- True positives: 360
- False positives: 268
- False negatives: 506
- Dataset-level **precision = 0.5732**, **recall = 0.4157**, **F1 = 0.4819**
- Mean per-image F1: 0.6533
- Median per-image F1: 0.7300
- Standard deviation of per-image F1: 0.2877

## Performance characterisation

The experiment is limited primarily by recall (the system under-predicts). The 0.29 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id  f1  tp  fp  fn
     101.png 1.0  10   0   1
    104.jpeg 1.0  11   0   7
     106.png 1.0   2   0   0
    110.jpeg 1.0   3   0   3
    111.jpeg 1.0   1   0   0

## Worst-performing images

    image_id   f1  tp  fp  fn
    112.jpeg 0.00   0   1   1
     78.jpeg 0.00   0   2   2
    118.jpeg 0.08   1  16  19
    119.jpeg 0.23   3  11  17
    121.jpeg 0.25   2   7  18
