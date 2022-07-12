# This just shows how to do logstic regression in pyspark
# I made this file because I'm trying to learn pyspark
# Note: it just runs on random data, so I didn't optimize any hyperparameters
import pyspark
from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler

from pathlib import Path
import os

from interfaces import IPySparkExample


class RegressionExample(IPySparkExample):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we want to run this in addition to the normal __init__ stuff
        self.set_save_name()

    def set_save_name(self, model_name: str="log_reg_model"):
        """
        Sets what the model will be saved as, when you run 'run_examples'
        This is also called in init

        Returns: None
        """

        save_to = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), # the folder of this script
            "model", # make a folder for models
            model_name
        )
        self._model_save_path = save_to

    def run_examples(self):
        # make 'features'
        # we just make the 'name_length' and 'random_val' columns
        # which are numeric values (yes they don't really make sense, this is just an example)
        sparkdf = self.create_sparkdf()
        sparkdf = self._get_name_num_letters(sparkdf)
        sparkdf = self._add_random_values(sparkdf)
        sparkdf = self._only_male_female(sparkdf)
        sparkdf = self._gender_to_boolean(sparkdf)

        # Next, we want to combine the different columns into ONE column
        # called 'features'. We can use the vectorassembler to do this.
        # Why: pyspark's linear regression only takes in one column, so it has
        # to be a 'combined' column with all the features. (I think so? Not 100% sure)
        sparkdf = self._use_vectorassembler(sparkdf)

        print("We can see that the 'features' column now exists, and it's full of vectors")
        sparkdf.show()

        # set up the logistic regression
        logreg = LogisticRegression(
            featuresCol="features",
            labelCol="gender"
        )

        # can set/view other settings here, like number of iterations.etc
        # I assume for other models types, it's the same
        logreg.setRegParam(0.5) # sets regularization strength, for linear regression

        # use the data to train the logistic regression
        # after fitting, this returns obj of type LogisticRegressionModel (not LogisticRegression)
        fitted_log_reg = logreg.fit(sparkdf)

        # Then, test the output on some dummy data
        testdata = self._make_test_data()
        testdata = self._use_vectorassembler(testdata) # don't forget to create 'features' column
        test_predictions = fitted_log_reg.transform(testdata)

        test_predictions.show()

        # Then, save the model to same directory
        print("Saving to: ", self._model_save_path)
        fitted_log_reg.write().overwrite().save(self._model_save_path)

        print("Finished saving model")


    @IPySparkExample._show_spark_df()
    def _get_name_num_letters(self, sparkdf: DataFrame):
        """
        Gets a new column, which is the number of letters in the name.
        """

        # you could also use expr, to do it in a sql-ish way
        sparkdf = sparkdf.withColumn(
            "name_length",
            f.length("first_name") + f.length("last_name")
        )

        return sparkdf

    @IPySparkExample._show_spark_df()
    def _add_random_values(self, sparkdf: DataFrame):
        "Creates 'random_val' column, which has uniform random dist"
        sparkdf = sparkdf.withColumn(
            "random_val",
            f.rand(420)
        )
        return sparkdf

    @IPySparkExample._show_spark_df()
    def _only_male_female(self, sparkdf: DataFrame):
        """
        Filters to only male/female people
        Reason is just cause there are the most of those users, and I want to make a logistic regression
        with only 2 values.
        """
        genders_to_keep = {"Male", "Female"}

        sparkdf = sparkdf.filter(
            f.col("gender").isin(genders_to_keep)
        )
        return sparkdf

    @IPySparkExample._show_spark_df()
    def _gender_to_boolean(self, sparkdf: DataFrame):
        """
        Right now the 'gender' column is 'Male' or 'Female', but we want to convert this
        into either 1 or 0.

        For this, we can just do it like SQL
        """
        
        sparkdf = sparkdf.withColumn(
            "gender",
            f.expr("""
            CASE
                WHEN gender = 'Male'
                THEN 1

                ELSE 0
            END
            """)
        )
        return sparkdf

    @IPySparkExample._show_spark_df()
    def _use_vectorassembler(self, sparkdf: DataFrame):
        """
        Uses the VectorAssembler to combine all columns we are interested in
        into one column (of type vector) called 'features'
        """

        # the column to output
        vec_assembler = VectorAssembler(outputCol="features")

        # the columns to use as input
        vec_assembler.setInputCols(["random_val", "name_length"])

        sparkdf = vec_assembler.transform(sparkdf)

        return sparkdf

    @IPySparkExample._show_spark_df()
    def _make_test_data(self):
        """
        Returns a DF with some made up values, so we can put it into
        the logistic regression and see what happens
        """

        columns = ["random_val", "name_length"]
        values =  [
            (0.123456, 35),
            (0.543210, 23)
        ]
        to_predict = self.spark.createDataFrame(
            values,
            columns
        )
        return to_predict


class LoadModelExample(IPySparkExample):
    "Loads the model we created earlier, and uses it"

    def run_examples(self) -> None:
        test_data = self._make_test_data()
        test_data = self._use_vectorassembler(test_data)

        fitted_log_reg = self._load_model()
        test_predictions = fitted_log_reg.transform(test_data)

        test_predictions.show()

        print("Done")

    def set_save_name(self, model_name: str="log_reg_model") -> None:
        """
        Tell this class where to load the model from

        Returns: None
        """

        save_to = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), # the folder of this script
            "model", # make a folder for models
            model_name
        )
        self._model_save_path = save_to

    @IPySparkExample._show_spark_df()
    def _make_test_data(self) -> DataFrame:
        """
        Returns a DF with some made up values
        """
        columns = ["random_val", "name_length"]
        values =  [
            (0.11111, 123),
            (0.22222, 456)
        ]
        to_predict = self.spark.createDataFrame(
            values,
            columns
        )
        return to_predict

    @IPySparkExample._show_spark_df()
    def _use_vectorassembler(self, sparkdf: DataFrame) -> DataFrame:
        """
        Uses the VectorAssembler to combine all columns we are interested in
        into one column (of type vector) called 'features'
        """
        # the column to output
        vec_assembler = VectorAssembler(outputCol="features")

        # the columns to use as input
        vec_assembler.setInputCols(["random_val", "name_length"])

        sparkdf = vec_assembler.transform(sparkdf)

        return sparkdf

    
    def _load_model(self) -> LogisticRegressionModel:
        """
        Returns: obj of type
        pyspark.ml.classification.LogisticRegressionModel
        
        LogisticRegression: before fitting
        LogisticRegressionModel: after fitting.

        In this case we already fitted it before, and we just load the
        model, so we want LogisticRegressionModel
        """
        fitted_log_reg = LogisticRegressionModel.load(self._model_save_path)
        return fitted_log_reg


    




if __name__ == "__main__":

    # make a model, test it, and save it
    regression_example = RegressionExample("dummySparkSession")
    regression_example.set_save_name("log_reg_model")
    regression_example.run_examples()

    # Load the model, and run it on some dummy data
    load_model_example = LoadModelExample("anotherSparkSession")
    load_model_example.set_save_name("log_reg_model")
    load_model_example.run_examples()