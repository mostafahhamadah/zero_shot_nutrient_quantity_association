# Dataset-level descriptive report — EXP-07

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 961
- True positives: 353
- False positives: 608
- False negatives: 513
- Dataset-level **precision = 0.3673**, **recall = 0.4076**, **F1 = 0.3864**
- Mean per-image F1: 0.5130
- Median per-image F1: 0.5900
- Standard deviation of per-image F1: 0.3507

## Performance characterisation

The experiment is limited primarily by precision and recall in roughly balanced proportion. The 0.35 standard deviation of per-image F1 indicates highly variable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id  f1  tp  fp  fn
       1.png 1.0   4   3   1
    109.jpeg 1.0   9   3   0
    110.jpeg 1.0   4   0   2
    116.jpeg 1.0  12   0   6
      15.png 1.0   9   0  33

## Worst-performing images

    image_id  f1  tp  fp  fn
     106.png 0.0   0   7   2
    111.jpeg 0.0   0   2   1
    112.jpeg 0.0   0   2   1
    118.jpeg 0.0   0  12  20
    119.jpeg 0.0   0  12  20
