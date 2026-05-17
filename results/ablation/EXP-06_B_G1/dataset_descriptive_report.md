# Dataset-level descriptive report — EXP-06

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 949
- True positives: 211
- False positives: 738
- False negatives: 655
- Dataset-level **precision = 0.2223**, **recall = 0.2436**, **F1 = 0.2325**
- Mean per-image F1: 0.3454
- Median per-image F1: 0.3600
- Standard deviation of per-image F1: 0.2824

## Performance characterisation

The experiment is limited primarily by precision and recall in roughly balanced proportion. The 0.28 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id   f1  tp  fp  fn
       1.png 1.00   4   3   1
       6.png 1.00   6   0   0
    109.jpeg 0.89   8   4   1
     101.png 0.82   9   3   2
      17.png 0.75   3   4   1

## Worst-performing images

    image_id  f1  tp  fp  fn
     106.png 0.0   0   7   2
    111.jpeg 0.0   0   2   1
    112.jpeg 0.0   0   2   1
    113.jpeg 0.0   0  22  20
    118.jpeg 0.0   0  11  20
