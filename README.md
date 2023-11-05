# sentiment_analysis_with_vader_and_roberta
Here you can see how to perform sentiment analysis via Roberta and Vader models. Roberta, developed by Hugging Face, and trained on Twitter data performs better than Vader. However, they are both poor in understanding sarcasm. A way to improve the model would be to train it specifically on Amazon data and using the reviews, in star, as a clue on the valence of the comments, by giving a high bias to 1 and 5 star reviews, a medium bias to 2 and 4 star reviews, and a neutral bias to 3 stars reviews.
When running the program, there a few packages you need to donlowad, they are marketd with as comments, to avoid downloading them at each trial, so uncomment them for the first time.
I suggest testing the code with only 55 rows first, so you can catch any bug, in the case that models and the code updated, and then you can expand the number of rows to 500 or even the whoel dataset if computing power is not a problem for you (e.g., for a full dataset, using a GPU instead of a CPU is required).
For this project, I want to credit Rob Mulla (https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/notebook) for the original code. I updated the code so it works with the latest updates in the packages involved for the sentiment analysis (especially the Vader model).
You can access the file used for this example here https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews.
