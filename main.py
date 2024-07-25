# Description: This file is the main file of the project.
from src.download_data import download_all
from src.data_preprocessing import data_preprocessing, data_integration
from src.visualization import visualization
from src.EDA import EDA
from src.data_inference import data_inference
from src.cloud_database import cloud_database
def main():
    #
    # acquire all of data we need and store them in csv files
    # here is an problem: yfinance sometimes doesn't work because of the protection of the website
    # if yfinance doesn't work,the program will automatically
    # use the already downloaded data of AAPL to continue the program.

    download_all()

    cloud_database()
    # # preprocess the data ,integrate it and store it in a csv file
    data_preprocessing()

    data_integration()

    #all pictures are stored in the "picture" folder
    visualization()

    #all pictures are stored in the "picture" folder
    EDA()

    #all pictures are stored in the "picture" folder
    data_inference()

    print("The project is done")











    """This method should be called when the program is run from the command line.
    The aim of the method is to run the complete, automated workflow you developed
    to solve the assignment.

    This function will be called by the automated test suite, so make sure that
    the function signature is not changed, and that it does not require any
    user input.

    If your workflow requires mongoDB (or any other) credentials, please commit them to
    this repository.
    Remember that if the workflow pushed new data to a mongo database without checking
    if the data is already present, the database will contain copies of the data and
    skew the results.

    After having implemented the method, please delete this docstring and replace
    it with a description of what your main method does.

    Hereafter, we provide a **volountarily suboptimal** example of how to structure
    your code. You are free to use this structure, and encouraged to improve it.

    Example:
        def main():
            # acquire the necessary data
            data = acquire()

            # store the data in MongoDB Atlas or Oracle APEX
            store(data)

            # format, project and clean the data
            proprocessed_data = preprocess(data)

            # perform exploratory data analysis
            statistics = explore(proprocessed_data)

            # show your findings
            visualise(statistics)

            # create a model and train it, visualise the results
            model = fit(proprocessed_data)
            visualise(model)
    """
   # raise NotImplementedError()


if __name__ == "__main__":
    main()

