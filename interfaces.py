# before running this, don't forget to conda activate pysparkexample
import pyspark

# These don't really work well with VScode (the autocomplete doesn't work properly?)
# so I just imported them directly
from pyspark.sql import functions as f
from pyspark.sql import DataFrame

from abc import ABC, abstractmethod
from functools import wraps

# TODO
# Group by and do some custom stuff
# make more than 1 dataframe, and join them?
# window functions
# How to use sklearn? Some kind of machine learning library. For now maybe just use Mllib



class IPySparkExample(ABC):
    """
    Just a base class that loads the data for you.

    Must provide session name.

    Then, call self.create_sparkdf() to get a spark DF
    """

    def __init__(self, app_name: str):
        """
        Params

        app_name: String
            As far as I can understand, this just lets you label the spark session.
            So if you provide a name like 'mysession' then you can go to the
            web UI and see that 'mysession' is taking up X amount of memory.etc
        """
        # Creates a 'session', which we will use for everything
        self.spark = (
            pyspark
            .sql
            .SparkSession
            .builder
            .appName(app_name)
            .getOrCreate()
        )

    def create_sparkdf(self):
        """
        Gets some data, creates a spark DF.
        The data is just fake data from https://www.mockaroo.com/
        and it's in a CSV (the CSV is not in git, I put it in .gitignore)
        """
        
        # How to specify delimiters and header/no header
        sparkdf = (
            self.spark.read
            .option("delimiter", ",")
            .option("header", "true")
            .csv("./data/MOCK_DATA.csv")
        )
        return sparkdf

    def _show_spark_df(numrows: int=5):
        """
        Decorator that shows some rows of any method that returns
        a spark DF.

        It will also print the class and method that it is decorating

        To use this in a child class just decorate with
        @IPySparkExample._show_spark_df()

        Default is 5 rows
        """
        # _show_spark_df is basically a factory function for decorators

        def mydeco(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                sparkdf = func(self, *args, **kwargs)
                print(str(type(self)), " | ", func.__name__)
                sparkdf.show(numrows)
                return sparkdf
            return wrapper
        return mydeco


    @abstractmethod
    def run_examples(self):
        "Call this to run the examples.etc"
        pass