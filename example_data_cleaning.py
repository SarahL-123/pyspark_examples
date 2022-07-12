import pyspark
from pyspark.sql import functions as f
from pyspark.sql import DataFrame

from interfaces import IPySparkExample

class DataCleaningExample(IPySparkExample):
        
    def run_examples(self):
        "Runs all the examples, printing something out each step"
        
        # Creates spark DF from CSV (you can also do it from pandas DF)
        sparkdf = self.create_sparkdf()
        
        # changes ID to integer (normally, don't do this! But just for our example)
        sparkdf = self._change_dtypes(sparkdf)

        # Adds values to spark DF
        self._add_rows(sparkdf)
        
        # Removes values from spark DF
        sparkdf = self._remove_rows(sparkdf)
        
        # this shows some stuff 
        self._byvalue_or_reference(sparkdf)
        
        # adds 123 as a column (as int)
        sparkdf = self._addcolumn_constant(sparkdf)
        
        sparkdf = self._udf_apply(sparkdf)
        
        sparkdf = self._udf_apply_multicolumn(sparkdf)

        self._summarystats(sparkdf)
        self._groupby_get_simple_stats(sparkdf)

        print("Done")

    @IPySparkExample._show_spark_df()
    def _change_dtypes(self, sparkdf: DataFrame):
        """
        The ID is a string
        We want to make it into an integer
        (normally you wouldn't do this, but I want to learn how to change column types)

        By the way you can see the dtypes by doing sparkdf.dtypes
        """

        sparkdf = sparkdf.withColumn(
            "id",
            f.col("id").cast('int')
        )

        print("Here are the new dtypes")
        print(sparkdf.dtypes)
        return sparkdf
    
    @IPySparkExample._show_spark_df()
    def _add_rows(
        self,
        sparkdf: DataFrame,
    ):
        """
        Add some rows to the dummy data

        To add rows, make another spark DF, then call the first DF's .union()
        method
        """
        
        # make another DF
        columns = ["id", "first_name", "last_name", "email", "gender", "ip_address"]
        values =  [
            ("-1", "Mister", "Bean",    "thebeanman@dummyaddress.com",         "Male",   "127.0.0.1"),
            ("-2", "Rick",   "Astley",  "nevergonnaemailyou@dummyaddress.com", "Male",   "127.0.0.2"),
            ("-3", "Remove", "Me",      "pleaseremoveme@dummyaddress.com",     "Female", "127.0.0.3"), # we will remove this later
            ("-4", "Remove", "Me2",     "alsoremoveme@dummyaddress.com",       "Male",   "127.0.0.4"), # we will remove this later
        ]
        newrows = self.spark.createDataFrame(
            values,
            columns
        )
        
        # combine the 2 DFs
        appended = sparkdf.union(newrows)

        return appended

    @IPySparkExample._show_spark_df()
    def _remove_rows(
        self,
        sparkdf: DataFrame, # check out type hints
    ):
        """
        To remove rows, use a filter
        """
        
        # using filter
        # (note: you can also use .where() instead of .filter(), it is
        #  exactly the same, it's just provided for people who like SQL)

        # This is one way to do it
        sparkdf = sparkdf.filter(
            "id != -3"
        )

        # This is another way to do it (seems safer, probably less vulnerable to injection??)
        # this is similar to pandas mydf=mydf[bool1 & bool2]
        sparkdf = sparkdf.filter(
            (f.col("id") != -4)
            & (f.col("id") != -3)
        )
        
        return sparkdf
    

    def _byvalue_or_reference(
        self,
        sparkdf: DataFrame
    ):
        """
        We see whether changing the DF without returning anything,
        affects the original dataframe (it doesn't)
        
        So it seems that the sparkdf object is just a 'window' into the 
        actual data, and we can have multiple windows
        """
        
        sparkdf.filter(
            "id != 1"
        )
        
    @IPySparkExample._show_spark_df()
    def _addcolumn_constant(
        self,
        sparkdf: DataFrame
    ):
        # How to add a constant value as a column
        sparkdf = sparkdf.withColumn(
            "dummy_value",
            f.lit(123) # lit means 'literal'
        )
        return sparkdf
        
    @IPySparkExample._show_spark_df()
    def _udf_apply(self, sparkdf: DataFrame):
        """
        A UDF (User Defined Function) is basically pandas apply, to each row.

        This is bad for performance, but I guess you can do anything with it, so
        I just put it here anyway.
        """
        
        # Let's make one that gets the first letter of their first_name.
        # First, we need to make a python function

        def get_first_letter(text: str):
            return text[0].upper()
        
        # next, we can make this python function into a UDF
        # However, we must specify the return type
        # This returns a udf object, which is like a special pyspark object that represents
        # a function in pyspark 
        get_first_letter_udf = f.udf(
            get_first_letter,
            pyspark.sql.types.StringType() # specify return type here
        )
        
        # (you can also use lambda functions)
        
        # Now we must actually use the UDF object
        # (this .withColumn method just lets you add a new column in general, not just
        # for UDFs. For example, earlier we added a constant as a column)
        sparkdf = sparkdf.withColumn(
            colName="first_name_letter", # the new column to create
            col=get_first_letter_udf("first_name") # we put 'first_name' into the UDF
        )
        
        return sparkdf
    
    @IPySparkExample._show_spark_df()
    def _udf_apply_multicolumn(self, sparkdf: DataFrame):
        """
        Another UDF, but this one uses multiple columns to create a new column.
        
        Returning multiple columns is apparently a bit complicated, so we won't
        do that for now.
        """
        
        def get_total_length(list_of_args: list):
            """
            This UDF/function should take in the id, first_name and last_name of each
            row, and return a number showing the total length of the name, plus
            the ID (which is an int)

            However, we will receive it as a list of arguments because I guess
            that's just how pyspark works?

            Please note: each entry in the list can have different dtypes.
            Here we are taking an int (id) and 2 strings (first and last name)
            """
            return (
                int(list_of_args[0])
                + len(list_of_args[1])
                + len(list_of_args[2])
            )
        
        total_len_udf = f.udf(
            get_total_length, # the function, which receives a LIST (which is one argument, same as before)
            pyspark.sql.types.IntegerType() # still returns an integer
        )
        
        sparkdf = sparkdf.withColumn(
            "total_len", # new column name
            
            # This time, we pass it not a column name, but a struct, which
            # is basically 'some columns'. Then, our function will get a list
            # (Notice that the first col is int and 2nd/3rd are strings, so we can
            # have different types)
            total_len_udf(
                f.struct("id", "first_name", "last_name")
            )
        )
        
        return sparkdf
    
    def _summarystats(self, sparkdf:DataFrame):
        """
        Gets the median value for some columns
        This is just the column we created in the previous method

        Returns: None (this prints stuff for you to read)
        """

        quantiles = sparkdf.approxQuantile(
            ["total_len", "id"], # this can be a string, or a list/tuple
            [0.1, 0.9], # quantiles (all values must be in [0, 1])
            relativeError=0.5
        )

        print("The quantiles are")
        print(quantiles)
    
    def _groupby_get_simple_stats(self, sparkdf: DataFrame):
        """
        Groups by a column, then gets summary stats for each group.

        Here, we group by the 'gender' column

        Returns: None (this prints stuff for you to read)
        """

        # After grouping, get several stats, including mean and quantile
        results_df = sparkdf.groupBy("gender").agg(
            f.mean("id").alias("id_mean"),

            # this basically lets you run statements like SQL
            f.expr("percentile_approx(total_len, 0.5) AS pctile") 
        )

        results_df.show()

if __name__ == '__main__':
    example = DataCleaningExample("myFirstSparkSession")
    example.run_examples()