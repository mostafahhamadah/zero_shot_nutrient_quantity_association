# Dataset-level descriptive report — EXP-08

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 610
- True positives: 355
- False positives: 255
- False negatives: 511
- Dataset-level **precision = 0.5820**, **recall = 0.4099**, **F1 = 0.4810**
- Mean per-image F1: 0.6689
- Median per-image F1: 0.7500
- Standard deviation of per-image F1: 0.2943

## Performance characterisation

The experiment is limited primarily by recall (the system under-predicts). The 0.29 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id  f1  tp  fp  fn
    104.jpeg 1.0  12   0   6
    111.jpeg 1.0   1   0   0
    112.jpeg 1.0   1   0   0
      12.png 1.0  12   0   4
      17.png 1.0   3   1   1

## Worst-performing images

    image_id   f1  tp  fp  fn
      30.png 0.00   0   8  18
     78.jpeg 0.00   0   2   2
    119.jpeg 0.07   1  16  19
      91.png 0.07   1  13  15
    118.jpeg 0.08   1  14  19
