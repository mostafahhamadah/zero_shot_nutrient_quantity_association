# Dataset-level descriptive report — EXP-01

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 878
- True positives: 156
- False positives: 722
- False negatives: 710
- Dataset-level **precision = 0.1777**, **recall = 0.1801**, **F1 = 0.1789**
- Mean per-image F1: 0.1925
- Median per-image F1: 0.0000
- Standard deviation of per-image F1: 0.2534

## Performance characterisation

The experiment is limited primarily by precision and recall in roughly balanced proportion. The 0.25 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id   f1  tp  fp  fn
    114.jpeg 0.89  16  10   4
    116.jpeg 0.86  12   6   6
    113.jpeg 0.71  10  12  10
    121.jpeg 0.67  12  15   8
      34.png 0.61  23  36  17

## Worst-performing images

    image_id  f1  tp  fp  fn
       1.png 0.0   0   3   5
     107.png 0.0   0  15  10
    108.jpeg 0.0   0  11  11
    111.jpeg 0.0   0   2   1
    112.jpeg 0.0   0   2   1
