# Dataset-level descriptive report — EXP-09

## Headline metrics

- Images processed: **57**
- Ground-truth tuples: 866
- Predicted tuples: 471
- True positives: 0
- False positives: 471
- False negatives: 866
- Dataset-level **precision = 0.0000**, **recall = 0.0000**, **F1 = 0.0000**
- Mean per-image F1: 0.0000
- Median per-image F1: 0.0000
- Standard deviation of per-image F1: 0.0000

## Performance characterisation

The experiment is limited primarily by precision and recall in roughly balanced proportion. The 0.00 standard deviation of per-image F1 indicates stable behaviour across images.

## Dominant failure sources

- **none**: 57 images

## Best-performing images

    image_id  f1  tp  fp  fn
       1.png 0.0   0   7   5
     101.png 0.0   0  11  11
     102.png 0.0   0  18  18
    103.jpeg 0.0   0  15  18
    104.jpeg 0.0   0  15  18

## Worst-performing images

    image_id  f1  tp  fp  fn
       1.png 0.0   0   7   5
     101.png 0.0   0  11  11
     102.png 0.0   0  18  18
    103.jpeg 0.0   0  15  18
    104.jpeg 0.0   0  15  18
