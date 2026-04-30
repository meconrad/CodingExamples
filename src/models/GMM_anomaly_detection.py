from CodingExamples.src.utils.modelling_utils import (
    identify_multi_modes_in_dataset,
    identify_anomalies_in_multimodal_data
)


class GMMAnomalyDetection():
    def __init__(
            self,
            data,
            min_scrub=0.05,
            max_scrub=99.5,
            max_components=5,
            max_iter=1000,
            ):
        """
        Scrubs dataset to specified percentile range (removing outliers), fits GMM to dataset to identify
        Gaussian components, calculates 3-sigma thresholds for each component, identifies anomalies outside
        of allowable 3-sigma bands, outputs anomalies detected into output dictionary.

        This class is assumed to be running in production; no need for data figures.

        Parameters
        ----------
        data : list
            Dataset to test with GMM

        min_scrub : float, default = 0.05
            minimum percentile to scrub outliers to, i.e. 0.03%

        max_scrub : float, default = 0.95
            maximum percentile to scrub outliers to, i.e. 99.7%

        max_components : int, default = 5
            Max components to test with GMM

        max_iter : int, default = 1000
            Max iterations to supply to GMM
    

        Returns
        -------
        output : dict
            {
                "thresholds": list of tuple,
                "n_anomalies": int,
                "anomaly_ratio": float (3 decimals),
                "anomalies": {
                                "date": int,
                                "value": float
                                }
                }
        """
        self.data = data
        self.min_scrub = min_scrub
        self.max_scrub = max_scrub
        self.max_components = max_components
        self.max_iter = max_iter

        self.get_component_thresholds()
        output = self.build_output()
        self.output_to_database() # Not yet implemented
        return output
    
    def get_component_thresholds(self):
        """
        Scrubs dataset to specified percentile range (removing outliers), fits GMM to dataset to identify
        Gaussian components, calculates 3-sigma thresholds for each component.
        """
        _, _, self.upper, self.lower, _ = identify_multi_modes_in_dataset(
            self.data,
            min_scrub=self.min_scrub,
            max_scrub=self.max_scrub,
            max_components=self.max_components,
            max_iter=self.max_iter,
            show_fig=False
        )

    def build_output(self):
        """
        Identifies anomalies outside of allowable 3-sigma bands, outputs anomalies detected into output
        dictionary.
        """
        output, _ = identify_anomalies_in_multimodal_data(
            self.data,
            self.lower,
            self.upper,
            show_fig=False
            )
        return output
    
    def output_to_database(self):
        """
        Output detected anomalies to noSQL database.
        """
        pass