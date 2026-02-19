from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FONT_SIZE_TICKS = 14
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16

def calculate_feature_importance(features,lr,X_test,y_test):  #X_test is the list of testing features
    if len(features)>0:
        bunch = permutation_importance(lr,X_test,y_test,n_repeats=10)   #here we are calculating the importance of each features shuffling each feature 10 times
        #features are included in X_test
        imp_means = bunch.importances_mean  #now these are the mean of importance of each feature
        ordered_imp_means = np.argsort(imp_means)[::-1]  #reversing the indices based on the value of importance mean and arsort() sorts the valeus but gives us the corresponding index
        result={}
        for i in ordered_imp_means:
            name = list(X_test.columns)[i] #this is the name of the feature obtained based on the index from ordered imp means
            importance_value = imp_means[i]  #the importance value of the current feature
            result.update({name:[importance_value]})
        most_important = list(X_test.columns)[ordered_imp_means[0]]
        result_df = pd.DataFrame.from_dict(result)  #converting the dictionary which consists of feature name and their corresponding importance into a dataframe
        return most_important,result_df

    else:
        return features[0],None


def plot_feature_importance(df):
    # Create a plot for feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Feature Name", fontsize=FONT_SIZE_AXES)
    ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)

    sns.barplot(data=df, orient="h", ax=ax, color="deepskyblue")

    plt.show()


def plot_happiness(variable, x1, y1, x2, y2):
    #here x1 and y1 are the training datasets and x2 and y2 are the testing datasets
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the original data
    ax.scatter(
        x1, y1, color="blue", edgecolors="white", s=15, label="Training Data"
    )
    # Plot the model
    ax.scatter(
        x2, y2,
        color="orange", edgecolors="black", s=15, marker="D", label="Predictions on the Test Set"
    )

    variable_title = " ".join(variable.split("_")).title()
    ax.set_xlabel(f"{variable_title}", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Life Lsadder (1-10)", fontsize=FONT_SIZE_AXES)
    ax.set_title(f"Happiness vs. {variable_title}", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.legend(fontsize=FONT_SIZE_TICKS)
