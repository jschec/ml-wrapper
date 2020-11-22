from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from sklearn.model_selection import train_test_split
from dataset import Dataset
from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

class IBuilder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @abstractproperty
    def product(self) -> None:
        pass

    @abstractmethod
    def produce_num_imputer_part(self) -> None:
        pass

    @abstractmethod
    def produce_num_scaler_part(self) -> None:
        pass

    @abstractmethod
    def produce_cat_imputer_part(self) -> None:
        pass

    @abstractmethod
    def produce_cat_encoder_part(self) -> None:
        pass

    @abstractmethod
    def produce_other_part(self) -> None:
        pass

# MAYBE CREATE BUILDERS FOR CLASSIFICATION VS CLUSTERING MODELS
# USE DIRECTOR TO COMPOSE STEPS

class ConcreteBuilder1(IBuilder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """
    def __init__(self, num_features: list, cat_features: list, target_feature: str) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        self.__num_features = num_features
        self.__cat_features = cat_features
        self.__target_feature = target_feature

        self.reset()

    def reset(self) -> None:
        self._product = Product1()

    #TODO
    @property
    def product(self) -> Product1:
        """
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different products that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another product.
        That's why it's a usual practice to call the reset method at the end of
        the `getProduct` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        """
        product = self._product
        self.reset()
        return product

    def configure_dataset(self, num_features: list, cat_features: list, src_data: any, target_feature: str) -> None:
        self.__dataset = Dataset(num_features, cat_features)
        self.__dataset.create_training_set(src_data, target_feature)
    
    def produce_num_imputer_part(self, sought_strat: str) -> None:
        self._product.num_step_add(imputing_strategy(sought_strat)) 

    def produce_num_scaler_part(self, sought_strat: str) -> None:
        self._product.num_step_add(scaling_strategy(sought_strat))

    def produce_cat_imputer_part(self, sought_strat: str) -> None:
        self._product.cat_step_add(imputing_strategy(sought_strat))

    def produce_cat_encoder_part(self, sought_strat: str) -> None:
        self._product.cat_step_add(encoding_strategy(sought_strat))
    
    def produce_other_part(self) -> None:
        self._other_product.add()

    def configure_model(self, model_type: str, model_name: str) -> None:
        if model_type == "classification":
            self._product.model_add(classification_models(model_name))
        elif model_type == "clustering":
            self._product.model_add(clustering_models(model_name))
        else:
            print("invalid option")

class ConcreteBuilder2(Builder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """

    def __init__(self) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        self.reset()

    def reset(self) -> None:
        self._product = Product1()
        
    @property
    def product(self) -> Product1:
        """
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different products that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another product.
        That's why it's a usual practice to call the reset method at the end of
        the `getProduct` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        """
        product = self._product
        self.reset()
        return product

    def produce_part_a(self) -> None:
        self._product.add("PartA1")

    def produce_part_b(self) -> None:
        self._product.add("PartB1")

    def produce_part_c(self) -> None:
        self._product.add("PartC1")

class Product1():
    """
    It makes sense to use the Builder pattern only when your products are quite
    complex and require extensive configuration.

    Unlike in other creational patterns, different concrete builders can produce
    unrelated products. In other words, results of various builders may not
    always follow the same interface.
    """

    def __init__(self) -> None:
        self.num_features = []
        self.cat_features = []
        self.target_feature = ""
        self.num_transformer_steps = []
        self.cat_transformer_steps = []
        self.decomposition_steps = []
        self.model_steps = []

    def num_transformer_add(self, transformer: Any) -> None:
        self.num_transformer_steps.append(transformer)

    def num_transformer_add_all(self, transformers: list) -> None:
        for transformer in transformers:
            self.num_transformer_add(transformer)

    def cat_transformer_add(self, transformer: Any) -> None:
        self.cat_transformer_steps.append(transformer)

    def cat_transformer_add_all(self, transformers: list) -> None:
        for transformer in transformers:
            self.cat_transformer_add(transformer)

    def decomposer_add(self, decomposer: Any) -> None:
        self.decomposition_steps.append(decomposer)
    
    def decomposer_add_all(self, decomposers: list) -> None:
        for decomposer in decomposers:
            self.decomposer_add(decomposer)

    def model_add(self, model: Any) -> None:
        self.model_steps.append(model)

    def model_add_all(self, models: Any) -> None:
        for model in models:
            self.model_add(model)

    def list_parts(self) -> None:
        print(f"Number transformer steps:   {', '.join(self.num_transformer_steps)}", end="")
        print(f"Categroy transformer steps: {', '.join(self.cat_transformer_steps)}", end="")
        print(f"Decomposition steps:        {', '.join(self.decomposition_steps)}", end="")
        print(f"Models:                     {', '.join(self.model_steps)}", end="")

    @property
    def preprocessor(self) -> ColumnTransformer:
        transformer_steps = []
        
        if len(self.num_transformer_steps) > 0:
            num_transformer = Pipeline(steps=self.num_transformer_steps)
            transformer_steps.append(('num', num_transformer,  self.num_features))
        

        if len(self.cat_transformer_steps) > 0:
            cat_transformer = Pipeline(steps=self.cat_transformer_steps)
            transformer_steps.append(('cat', cat_transformer, self.cat_features))

        preprocessor = ColumnTransformer(transformers=[
            transformer_steps
        ])

        pipeline_steps = [
            ('preprocessor', preprocessor)
        ]

        if len(self.decomposition_steps) > 0:
            for step in self.decomposition_steps:
                pipeline_steps.append(step)

        if len(self.model_steps) > 0:
            for step in self.model_steps:
                pipeline_steps.append(step)

        return pipeline_steps

class Director:
    """
    The Director is only responsible for executing the building steps in a
    particular sequence. It is helpful when producing products according to a
    specific order or configuration. Strictly speaking, the Director class is
    optional, since the client can control builders directly.
    """

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        """
        self._builder = builder

    """
    The Director can construct several product variations using the same
    building steps.
    """

    def build_minimal_viable_product(self) -> None:
        self.builder.produce_part_a()

    def build_full_featured_product(self) -> None:
        self.builder.produce_part_a()
        self.builder.produce_part_b()
        self.builder.produce_part_c()


    """
    TO ADD
    """

    def build_preprocessed_data(self, num_attrs: list =[], cat_attrs: list =[], decomp_steps: list =[]) -> None:
        self.builder.num_transformer_add_all(num_attrs)
        self.builder.cat_transformer_add_all(cat_attrs)

    def build_model(self,  model: object, num_attrs: list =[], cat_attrs: list =[]) -> None:
        self.builder.num_transformer_add_all(num_attrs)
        self.builder.cat_transformer_add_all(cat_attrs)
        self.builder.model_add(model)

    def build_models(self, models: list, num_attrs: list =[], cat_attrs: list =[]) -> None:
        self.builder.num_transformer_add_all(num_attrs)
        self.builder.cat_transformer_add_all(cat_attrs)
        self.builder.model_add_all(models)



def main(ml_dataset):
    """
    The client code creates a builder object, passes it to the director and then
    initiates the construction process. The end result is retrieved from the
    builder object.
    """

    experiment_config = json.loads("./config/experiment.json")
    dataset_config = experiment_config["dataset"]
    
    #numeric_features = experiment_config["dataset"]["numeric_features"]
    #categorical_features = experiment_config["dataset"]["categorical_features"]
    #target_feature = experiment_config["dataset"]["target_feature"]

    dataset = Dataset(config=dataset_config, data=ml_dataset)

    director = Director()
    builder = ConcreteBuilder1(dataset.num_features, dataset.cat_features, dataset.target_feature)
    director.builder = builder

    print("Standard basic product: ")
    director.build_minimal_viable_product()
    builder.product.list_parts()

    print("\n")

    print("Standard full featured product: ")
    director.build_full_featured_product()
    builder.product.list_parts()

    print("\n")

    # Remember, the Builder pattern can be used without a Director class.
    print("Custom product: ")
    builder.produce_part_a()
    builder.produce_part_b()
    builder.product.list_parts()

if __name__ == "__main__":
    main()