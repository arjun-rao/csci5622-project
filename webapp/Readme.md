
## How to use Bert model for predictions:

```
from bert import BertPredictor
model = BertPredictor()
model.load_model('model.load_model('../models_checkpoints/BertBest/best.pth')
result = model.predict('Perfection is not attainable , but if we chase perfection we can catch excellence .')
```