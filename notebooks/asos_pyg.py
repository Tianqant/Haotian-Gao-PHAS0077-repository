from typing import Callable, List, Optional
import os.path as osp
import torch
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_zip)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

country_mapping = {
    "Australia": 0,
    "Austria": 1,
    "Denmark": 2,
    "France": 3,
    "Germany": 4,
    "Netherlands": 5,
    "Sweden": 6,
    "UK": 7,
    "United States": 8
}

class ASOSData_pyg(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by ASOS AI Research from
    an internal dataset of purchases made on ASOS.com, consisting of nodes of
    type :obj:`"customer"` and :obj:`"product"`.
    Customer returns for products are available as ground truth labels for the edges
    between the customers and the products :obj:`("customer", "returns", "product")`.

    Args:
        load_path (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        model_name (str, optional): 
        reduced (bool, optional): 
        test (bool, optional): 
        include_product_nodes (bool, optional):
        include_country_nodes (bool, optional):
        
    """

    def __init__(self, load_path,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "asos-returns-model",
                 reduced: Optional[bool or int] = False,
                 test: Optional[bool] = False,
                 product_links: Optional[str or bool] = False,
                 country_links: Optional[str or bool] = False):

        # Loading in variables.
        self.name = "ASOS Data - pyg"
        self.load_path = load_path
        self.model_name = model_name
        self.test = test
        self.product_links = product_links if product_links == "virtual" or product_links == "direct" else False
        self.country_links = country_links if country_links == "virtual" or country_links == "direct" else False

        # Want the full test set so no reduced data allowed.
        if self.test:
            reduced = False

        # Define and create file paths for reduced or full datasets
        if not isinstance(reduced, bool):
            self.data_path = f"gnn_reduced_data"
            self.data_path_processed = f"data_node2vec_reduced_{self.model_name}_test.pt" if test else f"data_node2vec_reduced_{self.model_name}.pt"
            self.is_reduced = True

            path = os.path.join(self.load_path, "raw", self.data_path)
            if not os.path.exists(path):
                os.mkdir(path, mode=0o777)
                self.create_reduced_data(reduced)

        else:
            self.data_path = "gnn_reduced_data"
            self.data_path_processed = f"data_node2vec_{self.model_name}_test.pt" if test else f"data_node2vec_{self.model_name}.pt"
            self.is_reduced = False

        transform = None
        super().__init__(load_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Required by PyG
        """
        if self.test:
            return [
                osp.join(self.data_path, 'customer_nodes_training_FULL123.csv'),
                osp.join(self.data_path, 'product_nodes_training_FULL.csv'),
                osp.join(self.data_path, 'event_table_testing_FULL.csv'),
            ]
            # return [
            #     osp.join(self.data_path, 'top_customers_training.csv'),
            #     osp.join(self.data_path, 'top_products_training.csv'),
            #     osp.join(self.data_path, 'top_events_testing.csv')
            # ]
            # return [
            #     osp.join(self.data_path, 'customer_nodes_training_FULL.csv'),
            #     osp.join(self.data_path, 'product_nodes_training_FULL.csv'),
            #     osp.join(self.data_path, 'events_test.csv'),
            # ]
            # return [
            #     osp.join(self.data_path, 'customers_training_sample_v2.csv'),
            #     osp.join(self.data_path, 'products_training_sample_v2.csv'),
            #     osp.join(self.data_path, 'events_test.csv'),
            # ]
        else:
            return [
                osp.join(self.data_path, 'customer_nodes_training_FULL121k3m.csv'),
                osp.join(self.data_path, 'product_nodes_training_FULLdsdok.csv'),
                osp.join(self.data_path, 'event_table_training_FULL.csv'),
            ]
            # return [
            #     osp.join(self.data_path, 'top_customers_training.csv'),
            #     osp.join(self.data_path, 'top_products_training.csv'),
            #     osp.join(self.data_path, 'top_events_training.csv')
            # ]
            # return [
            #     osp.join(self.data_path, 'customer_nodes_training_FULL.csv'),
            #     osp.join(self.data_path, 'product_nodes_training_FULL.csv'),
            #     osp.join(self.data_path, 'events_train.csv'),
            # ]
            # return [
            #     osp.join(self.data_path, 'customers_training_sample_v2.csv'),
            #     osp.join(self.data_path, 'products_training_sample_v2.csv'),
            #     osp.join(self.data_path, 'events_train.csv'),
            # ]
    @property
    def processed_file_names(self) -> str:
        """
        Required by PyG
        """
        return self.data_path_processed

    def download(self):
        """
        Required by PyG
        """
        pass

    def describe(self):
        """
        Required by PyG
        """
        return self.data

    def create_reduced_data(self, reduced):
        """
        Method to sample and create the reduced dataset.
        """
        # Define the path to the full dataset
        full_path = os.path.join(self.load_path, "raw", "gnn_data")
        
        # Load in the customer node features - training set
        df_customers = pd.read_csv(os.path.join(full_path, "customer_nodes_training.csv"))
        df_products = pd.read_csv(os.path.join(full_path, "product_nodes_training.csv"))
        df_events = pd.read_csv(os.path.join(full_path, "event_table_training.csv"))

        # Load in the customer node features - test set
        df_customers_test = pd.read_csv(os.path.join(full_path, "customer_nodes_testing.csv"))
        df_products_test = pd.read_csv(os.path.join(full_path, "product_nodes_testing.csv"))
        df_events_test = pd.read_csv(os.path.join(full_path, "event_table_testing.csv"))

        # Check we have a large enough dataset to create a reduced set
        if len(df_events) > reduced:
            # Sample n events from full events (purchase) data
            df_events_reduced = df_events.sample(n=reduced, random_state=np.random.get_state()[1][0])
        else:
            df_events_reduced = df_events

        # Only keep the customers and products we have events for in the customer and product data
        df_products_reduced = pd.merge(df_events_reduced["variantID"], df_products, 
                                            on="variantID", how="inner").drop_duplicates()
        df_customers_reduced = pd.merge(df_events_reduced["hash(customerId)"], df_customers, 
                                            on="hash(customerId)", how="inner").drop_duplicates()

        # Create file path for reduced data.
        full_path_reduced = os.path.join(self.load_path, "raw", self.data_path)

        # Save all new reduced sets to folder
        df_customers_reduced.to_csv(os.path.join(full_path_reduced, "customer_nodes_training.csv"), index=False)
        df_products_reduced.to_csv(os.path.join(full_path_reduced, "product_nodes_training.csv"), index=False)
        
        df_customers_test.to_csv(os.path.join(full_path_reduced, "customer_nodes_testing.csv"), index=False)
        df_products_test.to_csv(os.path.join(full_path_reduced, "product_nodes_testing.csv"), index=False)

        df_events_reduced.to_csv(os.path.join(full_path_reduced, "event_table_training.csv"), index=False)
        df_events_test.to_csv(os.path.join(full_path_reduced, "event_table_testing.csv"), index=False)
        
    def add_variant_product_links(self, data, products, link_type):
        """
        Method for finding the variant-product links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the variant-product links.
        variant_product_links = pd.read_csv(os.path.join(self.load_path, "raw", "gnn_reduced_data", "variant_table_training.csv"))
        
        # Find all unique products and label the products dataset with indices from 0 to N (number of products)
        # This is necessary as productID is not a linear variable.
        product_mapping = pd.DataFrame(products["productID"].unique()).reset_index().set_index(0).to_dict()["index"]
        products["product_id"] = products["productID"].map(product_mapping)
        
        # Create a table of variants labelled with their product IDs.
        valid_variant_product_links = variant_product_links.merge(
                                        products[["variantID", "variant_id", "productID", "product_id"]], 
                                        on=["variantID", "productID"], how="inner")

        if link_type == "virtual":
            # Find the source (variant) and destination (product) nodes
            variant_src = torch.tensor(valid_variant_product_links["variant_id"].values)
            product_dst = torch.tensor(valid_variant_product_links["product_id"].values)
            
            # Create the edges.
            var_prod_edge = torch.stack([variant_src, product_dst])
        
            # Get virtual node information (this is currently just averages but could be adapted)
            avg_product_vals = products.drop(["variantID", 
                                          "productID", 
                                          "variant_id"], axis=1).groupby("product_id").mean()
        
            # Add node specific information to the graph product nodes
            data["product"].x = torch.from_numpy(avg_product_vals.to_numpy()).to(torch.float)
            
            # Get a list of indicies for product nodes
            product_nodes = list(range(0, len(avg_product_vals.index.unique())))
        
            data["product"].num_nodes = int(len(product_nodes))
            data['product'].node_index = torch.tensor(product_nodes)
            
            # Add the edges (and reverse edges) to the graph structure
            data['variant', 'belongs_to', 'product'].edge_index = var_prod_edge.to(torch.long)
            data['product', 'includes', 'variant'].edge_index = torch.flip(var_prod_edge.to(torch.long), [0])
            
        elif link_type == "direct":
            # ADD CODE HERE (and delete pass)
            pass
        else:
            pass
        
    def add_customer_country_links(self, data, customers, link_type):
        """
        Method for finding the customer-country links and incorporating these into the graph structure.
        Either by using virtual nodes or by adding direct links.
        """
        # Load in the customer-country links.
        customer_country_links = pd.read_csv(os.path.join(self.load_path, "raw", "gnn_reduced_data", "country_table_training.csv"))
        
        # Create a table of customers labelled with their country IDs.
        valid_customer_country_links = customer_country_links.merge(
                                        customers[["customer_id", "hash(customerId)"]], 
                                        on=["hash(customerId)"], how="inner")

        if link_type == "virtual":
            # Find the source (customer) and destination (country) nodes
            customer_src = torch.tensor(valid_customer_country_links["customer_id"].values)
            country_dst = torch.tensor(valid_customer_country_links["countryID"].values)
            
            # Create the edges.
            cus_country_edge = torch.stack([customer_src, country_dst])
            
            # Delete country information as this is now encoded into the graph
            filter_col = [col for col in customers.columns if col.startswith("country_")]
            customers.drop(filter_col, axis=1, inplace=True)
        
            # Get virtual node information (this is currently just averages but could be adapted)
            avg_country_vals = customers.drop(["hash(customerId)", 
                                               "customer_id"], axis=1).groupby("shippingCountry").mean()
        
            # Add node specific information to the graph country nodes
            data["country"].x = torch.from_numpy(avg_country_vals.to_numpy()).to(torch.float)
            
            # Get a list of indicies for country nodes
            country_nodes = list(range(0, len(avg_country_vals.index.unique())))
        
            data["country"].num_nodes = int(len(country_nodes))
            data['country'].node_index = torch.tensor(country_nodes)
            
            # Add the edges (and reverse edges) to the graph structure
            data['customer', 'is_from', 'country'].edge_index = cus_country_edge.to(torch.long)
            data['country', 'from_is', 'customer'].edge_index = torch.flip(cus_country_edge.to(torch.long), [0])
            
        elif link_type == "direct":
            # ADD CODE HERE (and delete pass)
            pass
            
        else:
            pass

    def process(self):
        """
        Process method to construct the graph dataset.
        """
        # Define data variable
        data = HeteroData()

        # Read in customer information and perform transformations on this
        df_customers = pd.read_csv(self.raw_paths[0]).dropna()

        # Insert a linear customer ID variable (required for labelling nodes)
        df_customers.insert(0, "customer_id", range(0, len(df_customers)))
        
        # Read in and transform product information
        df_products = pd.read_csv(self.raw_paths[1]).dropna()
        
        # Insert a linear product ID variable (required for labelling nodes)
        df_products.insert(0, "variant_id", range(0, len(df_products)))
        
        # Read in the purchase links and merge with above datasets to ensure 
        # node IDs are included (customer_id -> variant_id)
        df_events = pd.read_csv(self.raw_paths[2]).dropna()
        
        df_events = df_events.merge(df_customers[["hash(customerId)", "customer_id"]], 
                                    on="hash(customerId)", how="inner")

        df_valid_events = df_events.merge(df_products[["variantID", "variant_id"]], 
                                          on="variantID", how="inner")
                
        # Construct the purchase links
        customer_src = torch.tensor(df_valid_events["customer_id"])
        product_dst = torch.tensor(df_valid_events["variant_id"])
        edge_index = torch.stack([customer_src, product_dst])

        # Get the labels for these purchase links
        returned = torch.from_numpy(df_valid_events['isReturned'].values).to(torch.bool)
        
        # Get the indicies for return edges
        return_edge_index = edge_index[:,returned]
        
        # Add extra links
        if self.product_links:
            self.add_variant_product_links(data, df_products, link_type=self.product_links)
            
        if self.country_links:
            self.add_customer_country_links(data, df_customers, link_type=self.country_links)
        
        # Removes ids from node information, add it to index instead.
        df_customers = df_customers.set_index("customer_id")
        df_products = df_products.set_index("variant_id")
        
        # Drop non-useful information
        df_customers.drop(["hash(customerId)", "shippingCountry"], axis=1, inplace=True)
        df_products.drop(["variantID", "productID", "brandDesc", "productType"], axis=1, inplace=True)

        # Add node features for customers and variants
        data['customer'].x = torch.from_numpy(df_customers.to_numpy()).to(torch.float)
        data['variant'].x = torch.from_numpy(df_products.to_numpy()).to(torch.float)
        
        # Add purchase links and labels for these
        data['customer', 'purchases', 'variant'].edge_index = edge_index.to(torch.long)
        data['customer', 'purchases', 'variant'].edge_label = returned.to(torch.long)
        data['variant', 'purchased_by', 'customer'].edge_index = torch.flip(edge_index.to(torch.long), [0])
        
        # create the edge of "customer - returns- product" for both train and test
        data['customer', 'returns', 'variant'].edge_index = return_edge_index.to(torch.long)
        data['variant', 'returned_by', 'customer'].edge_index = torch.flip(return_edge_index.to(torch.long), [0])

        # Add node information for graph
        customer_nodes = int(len(df_customers))
        product_nodes = int(len(df_products))
        data['customer'].num_nodes = customer_nodes
        data['variant'].num_nodes = product_nodes
        
        data['customer'].node_index = torch.tensor(df_customers.index)
        data['variant'].node_index = torch.tensor(df_products.index)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])