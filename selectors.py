import config_models
import config_strategies

class Selector:
    "Container for available items"

    #private members
    __items = {}
    __selector_type = ""

    def __init__(self, selector_type, items):
        """
        Summary line.

        Constructor for selector class

        Parameters
        ----------
        self : TODO
            TODO
        selector_type : str
            TODO
        items : dict
            TODO
        """
        self.__selector_type = selector_type
        self.__items = items
    
    def get_item(self, item_name):
        """
        Summary line.

        Retrieves the sought element

        Parameters
        ----------
        item_name : str
            Name of the element to retrieve
        
        Returns
        -------
        method
            Sought pythonic function

        """
        selected_item = self.items.get(item_name, lambda: f"Invalid {self.selector_type} strategy name")
        return (self.__selector_type, selected_item)

    def available_items(self):
        """
        Summary line.

        Prints out available elements in this selector

        Parameters
        ----------
        item_name : str
            Description of arg1
        
        """
        counter = 0
        for key in self.items:
            print(f"{counter} : {key}")
            counter += 1


class encoding_strategy(Selector):
    """
    Class for all available encoding strategies for sklearn pipeline transformations, in which a strategy
    """
    def __init__(self):
        super.__init__("encoding", config_strategies.encoding_strats)

class scaling_strategy(Selector):
    """
    Class for all available scaling strategies sklearn pipeline transformations
    """
    def __init__(self):
        super.__init__("scaling", config_strategies.scaling_strats)

class imputing_strategy(Selector):
    """
    Class for all available imputing strategies sklearn pipeline transformations
    """
    def __init__(self):
        super.__init__("imputing", config_strategies.inmputing_strats)

class decomposition_strategy(Selector):
    """
    
    """
    def __init__(self):
        super.__init__("imputing", config_strategies.decomposition_strats)

class feature_Selector(Selector):
    """
    Class for all available feature selection strategies to be placed in sklearn pipeline following data transformations
    """
    def __init__(self):
        super.__init__("clustering", config_strategies.feature_selection_strats)

class classification_ml_model(Selector):
    """
    Class for all available classification machine learning models
    """
    def __init__(self):
        super.__init__("classification", config_models.classification_models)

class clustering_ml_model(Selector):
    """
    Class for all available clustering machine learning models
    """
    def __init__(self):
        super.__init__("clustering", config_models.clustering_models)
