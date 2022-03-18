# Repository for experiments with TLP
**All scripts have to be executed from root directory!!!**

### How to download datasets
Because there are *.dvc files in root directory we can download them using this command:
```shell
dvc pull
```

## OpenAI embeddings:
The main goal is to create embeddings using [OpenAI API](https://beta.openai.com/docs/guides/embeddings) to represent code snippets with vectors, then pass them to simple linear classifier and compare results with our classification model.
### Results:
After running few iterations on toy dataset, we noticed that price is higher than we expected. It turned out that the actual price is much higher, and it is not worth it to continue this experiment.

## OCR comparison:
The main competitor is [APITemple](https://apilayer.com/marketplace/description/image_to_text-api), because it is also offered by [APILayer](https://apilayer.com/). Our goal is to run our dataset through their API and compare the accuracy with our current OCR model.
### Dataset:
We decided to use dataset built from manually labelled code screenshots, because it is the most representative.
### Results:
Using their API is really painful, we had to manually run our dataset through [linkpicture](https://www.linkpicture.com/en/?set=en) to get URL of every image, so we could pass them in request in python. Example of better input handling is [Microsoft's OCR API](https://westus.dev.cognitive.microsoft.com/docs/services/57cf753a3f9b070c105bd2c1/operations/57cf753a3f9b070868a1f66b).

## Owner:
Patryk Bartkowiak