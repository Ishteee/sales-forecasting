import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Initialize session state
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'X_test_original' not in st.session_state:
    st.session_state.X_test_original = None


def show_welcome():
    st.title("Welcome to our Sales Forecasting App!")
    if st.button("Continue"):
        st.session_state.algorithm = "Upload CSV"

    st.markdown("""
    <style>
    .bottom-right {
        position: fixed; /* Fixed position */
        bottom: 80px; /* Distance from the bottom */
        left: 450px; /* Distance from the right */
        font-size: 23px; /* Font size */
        color: #fff; /* Text color */
        background-color: rgba(70, 70, 70, 0.2); /* Background color with opacity */
        padding: 40px; /* Padding */
        border-radius: 5px; /* Rounded corners */
        margin: 70px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Optional shadow */
    }
    </style>
    <div class="bottom-right">
        Istiyaq Chawdhary - 225035 <br>
        Feba Thomas - 225015 <br>
        Gaurav Chaudhary - 225047
    </div>
""", unsafe_allow_html=True)
    
def show_upload_csvs():
    st.title("Upload CSV Files")
    st.write("Please upload all required CSV files:")

    train_file = st.file_uploader("Upload Train CSV", type=["csv"])
    test_file = st.file_uploader("Upload Test CSV", type=["csv"])
    oil_file = st.file_uploader("Upload Oil CSV", type=["csv"])
    holidays_file = st.file_uploader("Upload Holidays and Events CSV", type=["csv"])
    stores_file = st.file_uploader("Upload Stores CSV", type=["csv"])

    if train_file and test_file and oil_file and holidays_file and stores_file:
        st.session_state.train_data = pd.read_csv(train_file)
        st.session_state.test_data = pd.read_csv(test_file)
        st.session_state.oil_data = pd.read_csv(oil_file)
        st.session_state.holidays_data = pd.read_csv(holidays_file)
        st.session_state.stores_data = pd.read_csv(stores_file)

        if st.button("Proceed to Algorithm Selection"):
            st.session_state.algorithm = "Algorithm Selection"


def show_algorithm_selection():
    st.title("Select an Algorithm")
    st.write("Which Algorithm would you like to use?")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Random Forest"):
            st.session_state.algorithm = "Random Forest"
            st.session_state.step = 3

    with col2:
        if st.button("Neural Network"):
            st.session_state.algorithm = "Neural Network"
            st.session_state.step = 3

    with col3:
        if st.button("XGradient Boost"):
            st.session_state.algorithm = "XGradient Boost"
            st.session_state.step = 3

def show_predictions():
    st.title(f"Running {st.session_state.algorithm} Predictions")

    train = st.session_state.train_data
    test = st.session_state.test_data
    oil = st.session_state.oil_data
    holidays = st.session_state.holidays_data

    def date_type(df):
        '''change the date column in a data frame to datetime'''
        df = df.copy()
        df['date']= pd.to_datetime(df['date'])
        return df

    train=date_type(train)
    test=date_type(test)

    def extract_datetime_features(df):
        '''Extracting some datetime features 
        like year, month, day of month, and day of week'''
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dayOfMonth'] = df['date'].dt.day
        df['dayOfWeek'] = df['date'].dt.dayofweek
        return df

    train=extract_datetime_features(train)
    test=extract_datetime_features(test)

    df=pd.concat([train, test]).reset_index(drop=True)

    holidays=date_type(holidays)
    oil=date_type(oil)

    holidays.drop(columns=['description','locale_name'], inplace=True)

    holidays.isna().sum()

    oil.isna().sum()



    df = df.merge(holidays,how='left',on='date')
    df = df.merge(oil,how='left',on='date')

    def get_unique(df,column_name):
        '''Get the all values and the count for specific column'''
        unique_values_count = df[column_name].nunique()
        unique_values = df[column_name].unique()

        print(f"Number of unique values in {column_name}: {unique_values_count}")
        print("Unique values:")
        for value in unique_values:
            print(value)

    df = pd.get_dummies(df, columns=['family'], dummy_na=False, prefix='family')
    df = pd.get_dummies(df, columns=['type'], dummy_na=False, prefix='holidayType')
    df = pd.get_dummies(df, columns=['locale'], dummy_na=False, prefix='holidayLocale')
    df = pd.get_dummies(df, columns=['transferred'], dummy_na=False, prefix='holidayTransferred')

    df.isna().sum()

    new_oil=df[['date','dcoilwtico']].drop_duplicates(subset='date', keep="first").reset_index(drop=True)
    print(new_oil.shape)
    print(new_oil.isna().sum())

    new_oil['dcoilwtico'].fillna(method='ffill', inplace=True)
    new_oil['dcoilwtico'].fillna(method='bfill', inplace=True)


    df=df.drop(columns=['dcoilwtico']) # remove the old column from df before merging with the new_oil
    df = df.merge(new_oil,how='left',on='date')

    print(df.isna().sum())

    print(f'Before removing duplicates {df.shape}')

    boolean_columns = [
    'holidayType_Additional',
    'holidayType_Holiday',
    'holidayType_Transfer',
    'holidayLocale_Local',
    'holidayLocale_National',
    'holidayTransferred_False',
    'holidayTransferred_True'
    ]

    # Group by 'id' and take the union of the boolean columns
    df_grouped = df.groupby('id')[boolean_columns].max().reset_index()

    # Take the first occurrence of non-boolean columns
    non_boolean_columns = [col for col in df.columns if col not in boolean_columns]
    non_boolean_columns.remove('id')
    df_non_boolean = df.groupby('id')[non_boolean_columns].first().reset_index()

    # Merge the non-boolean and boolean DataFrames
    df = pd.merge(df_non_boolean, df_grouped, on='id')

    print(f'After removing duplicates {df.shape}')

    train = df[df['date']<='15-08-2017'].reset_index(drop=True)
    test = df[df['date']>'15-08-2017'].reset_index(drop=True)

    # Drop the 'id' column
    train.drop('id', axis=1, inplace=True)

    # Define the date for the split
    split_date = '2017-08-01' # we got 15 days only for val since our test set is 16 days only

    # Create training and validation sets
    train_set = train[train['date'] < split_date]
    val_set = train[train['date'] >= split_date]

    train_set.drop('date', axis=1, inplace=True)
    val_set.drop('date', axis=1, inplace=True)

    # Define X and y for training and validation
    X_train = train_set.drop('sales', axis=1)  # Features for training
    y_train = train_set['sales']  # Target for training
    X_val = val_set.drop('sales', axis=1)  # Features for validation
    y_val = val_set['sales']  # Target for validation

    test_set_date = test.copy().drop('sales', axis=1)

    X_test=test.drop(['id','sales','date'], axis=1)

    from sklearn.preprocessing import StandardScaler

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on X_train and transform X_train
    X_train = scaler.fit_transform(X_train)

    # Use the same scaler to transform X_val and X_test
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if st.session_state.algorithm == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        # Ensure that the data types are appropriate (e.g., float32)
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        X_test=X_test.astype(np.float32)

        # Initialize Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model on training data
        rf_model.fit(X_train, y_train)

        # Predict on validation data
        y_val_pred_rf = rf_model.predict(X_val)

        # Calculate and display the MSE for validation set predictions
        val_mse_rf = mean_squared_error(y_val, y_val_pred_rf)
        st.write(f"Random Forest Validation MSE: {val_mse_rf}")

        # Predict on the test set
        y_test_pred_rf = rf_model.predict(X_test)

        # Create a new dataframe with id and sales predictions from Random Forest
        rf_predictions_df = pd.DataFrame({'id': test['id'], 'date': pd.to_datetime(test['date']).dt.date, 'sales': y_test_pred_rf.flatten()})

        # Ensure non-negative sales predictions (adjust as needed)
        rf_predictions_df['sales'] = rf_predictions_df['sales'].clip(lower=0)

        # Display the Random Forest predictions dataframe
        st.write("Random Forest Sales Predictions")

        # Display the families with the most sales
        test = test.drop(columns=['sales'])
        test = test.merge(rf_predictions_df[['id', 'sales']], on='id', how='left')

        # Filter out family columns and sales column
        family_columns = [col for col in test.columns if col.startswith('family_')]
        sales_column = 'sales'

        # Calculate the total sales for each family
        family_sales = {}
        for family in family_columns:
            family_sales[family] = test.loc[test[family] == 1, sales_column].sum()

        # Convert to DataFrame
        family_sales_df = pd.DataFrame(list(family_sales.items()), columns=['Family', 'Total Sales'])

        
        # Plotting
        st.title("Total Predicted Sales by Product Family")

        st.write(family_sales_df)

        # Create a bar plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total Sales', y='Family', data=family_sales_df, palette='viridis')

        # Display the plot
        st.pyplot(plt)



        # Display the most sales by store number
        store_sales = test.groupby('store_nbr')['sales'].sum().reset_index()

        # Sort the stores by sales
        store_sales = store_sales.sort_values(by='sales', ascending=False)

        # Streamlit app to show the bar chart
        st.title('Predicted Sales by Store')

        st.write('Predicted Sales data by store number:')
        st.dataframe(store_sales)

        # Plot the bar chart
        fig, ax = plt.subplots()
        ax.bar(store_sales['store_nbr'], store_sales['sales'])
        ax.set_xlabel('Store Number')
        ax.set_ylabel('Total Sales')
        ax.set_title('Total Sales by Store Number')

        # Display the chart in Streamlit
        st.pyplot(fig) 





        # Display Predicted Sales by City
        df_store_details = st.session_state.stores_data  # Store details with store_nbr, city, etc.

        # Filter out necessary columns for efficient lookup
        df_store_details = df_store_details[['store_nbr', 'city']]  # Only store_nbr and city columns

        # Group sales by store_nbr
        sales_by_store = test.groupby('store_nbr')['sales'].sum().reset_index()

        # Now link the two dataframes based on store_nbr (without full merge)
        # We map the city for each store_nbr in the sales_by_store DataFrame
        sales_by_store['city'] = sales_by_store['store_nbr'].map(df_store_details.set_index('store_nbr')['city'])

        # Group sales by city
        sales_by_city = sales_by_store.groupby('city')['sales'].sum().reset_index()

        # Sort by sales to find the cities with highest sales
        sales_by_city = sales_by_city.sort_values(by='sales', ascending=False)

        # Streamlit app to display the data and plot
        st.title('Sales by City')

        st.write('Total sales data grouped by city:')
        st.dataframe(sales_by_city)


        # Create a bar plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='sales', y='city', data=sales_by_city, palette='viridis')

        # Set custom labels
        ax.set_xlabel('Predicted Sales (in millions)')  # Custom x-axis label
        ax.set_ylabel('City')         # Custom y-axis label
        ax.set_title('Sales by City') # Custom title

        # Display the plot
        st.pyplot(plt)


        test_set_date=test_set_date.merge(rf_predictions_df,how='left',on='id')

        # Merge the Random Forest predictions with the test set for plotting
        test_set_date_rf = test_set_date.copy()
        test_set_date_rf = test_set_date_rf.merge(rf_predictions_df, how='left', on='id', suffixes=('', '_rf'))

        # Plot sales predictions from the Random Forest model
        st.title("Predicted Sales Over Time using Random Forest")

        st.write(rf_predictions_df)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the "train" dataframe in blue
        ax.scatter(train['date'], train['sales'], label='Train Sales', marker='o', s=10, c='blue')

        # Plot the Random Forest predictions in green
        ax.scatter(test_set_date_rf['date'], test_set_date_rf['sales_rf'], label='Random Forest Test Sales', marker='o', s=10, c='green')

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Predicted Sales by Date - Random Forest')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        if st.button("Back to Algorithm Selection"):
            st.session_state.algorithm = "Algorithm Selection"

    elif st.session_state.algorithm == "Neural Network":
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Ensure that the data types are appropriate (e.g., float32)
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        # Define the deep learning model
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),  # Notice the extra parentheses around shape
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        # Compile the model with a suitable loss function and optimizer
        model.compile(optimizer='adam', loss='mean_squared_error')  # You can adjust the loss function

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, verbose=1)

        # 1. Make predictions on X_test

        X_test=X_test.astype(np.float32)
        y_test_pred = model.predict(X_test)

        # 2. Create a new dataframe with id and sales predictions

        # Assuming test dataframe has an 'id' column
        predictions_df = pd.DataFrame({'id': test['id'], 'date': pd.to_datetime(test['date']).dt.date, 'sales': y_test_pred.flatten()})

        # 3. Optionally, ensure non-negative sales predictions (adjust as needed)
        predictions_df['sales'] = predictions_df['sales'].clip(lower=0)  # Clip negative values to 0

        # Display the predictions dataframe
        print(predictions_df)

        # Display the XGBoost predictions dataframe
        st.write("Neural Network (DNN) Sales Predictions")












        # Display the families with the most sales
        test = test.drop(columns=['sales'])
        test = test.merge(predictions_df[['id', 'sales']], on='id', how='left')

        # Filter out family columns and sales column
        family_columns = [col for col in test.columns if col.startswith('family_')]
        sales_column = 'sales'

        # Calculate the total sales for each family
        family_sales = {}
        for family in family_columns:
            family_sales[family] = test.loc[test[family] == 1, sales_column].sum()

        # Convert to DataFrame
        family_sales_df = pd.DataFrame(list(family_sales.items()), columns=['Family', 'Total Sales'])

        
        # Plotting
        st.title("Total Predicted Sales by Product Family")

        st.write(family_sales_df)

        # Create a bar plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total Sales', y='Family', data=family_sales_df, palette='viridis')

        # Display the plot
        st.pyplot(plt)



        # Display the most sales by store number
        store_sales = test.groupby('store_nbr')['sales'].sum().reset_index()

        # Sort the stores by sales
        store_sales = store_sales.sort_values(by='sales', ascending=False)

        # Streamlit app to show the bar chart
        st.title('Predicted Sales by Store')

        st.write('Predicted Sales data by store number:')
        st.dataframe(store_sales)

        # Plot the bar chart
        fig, ax = plt.subplots()
        ax.bar(store_sales['store_nbr'], store_sales['sales'])
        ax.set_xlabel('Store Number')
        ax.set_ylabel('Total Sales')
        ax.set_title('Total Sales by Store Number')

        # Display the chart in Streamlit
        st.pyplot(fig) 





        # Display Predicted Sales by City
        df_store_details = st.session_state.stores_data  # Store details with store_nbr, city, etc.

        # Filter out necessary columns for efficient lookup
        df_store_details = df_store_details[['store_nbr', 'city']]  # Only store_nbr and city columns

        # Group sales by store_nbr
        sales_by_store = test.groupby('store_nbr')['sales'].sum().reset_index()

        # Now link the two dataframes based on store_nbr (without full merge)
        # We map the city for each store_nbr in the sales_by_store DataFrame
        sales_by_store['city'] = sales_by_store['store_nbr'].map(df_store_details.set_index('store_nbr')['city'])

        # Group sales by city
        sales_by_city = sales_by_store.groupby('city')['sales'].sum().reset_index()

        # Sort by sales to find the cities with highest sales
        sales_by_city = sales_by_city.sort_values(by='sales', ascending=False)

        # Streamlit app to display the data and plot
        st.title('Sales by City')

        st.write('Total sales data grouped by city:')
        st.dataframe(sales_by_city)


        # Create a bar plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='sales', y='city', data=sales_by_city, palette='viridis')

        # Set custom labels
        ax.set_xlabel('Predicted Sales (in millions)')  # Custom x-axis label
        ax.set_ylabel('City')         # Custom y-axis label
        ax.set_title('Sales by City') # Custom title

        # Display the plot
        st.pyplot(plt)


        # test_set_date=test_set_date.merge(rf_predictions_df,how='left',on='id')

        # # Merge the Random Forest predictions with the test set for plotting
        # test_set_date_rf = test_set_date.copy()
        # test_set_date_rf = test_set_date_rf.merge(rf_predictions_df, how='left', on='id', suffixes=('', '_rf'))

        # # Plot sales predictions from the Random Forest model
        # st.title("Predicted Sales Over Time using Random Forest")

        # st.write(rf_predictions_df)

        # fig, ax = plt.subplots(figsize=(12, 6))

        # # Plot the "train" dataframe in blue
        # ax.scatter(train['date'], train['sales'], label='Train Sales', marker='o', s=10, c='blue')

        # # Plot the Random Forest predictions in green
        # ax.scatter(test_set_date_rf['date'], test_set_date_rf['sales_rf'], label='Random Forest Test Sales', marker='o', s=10, c='green')

        # ax.set_xlabel('Date')
        # ax.set_ylabel('Sales')
        # ax.set_title('Predicted Sales by Date - Random Forest')
        # ax.legend()
        # ax.grid(True)
        # st.pyplot(fig)























        test_set_date=test_set_date.merge(predictions_df,how='left',on='id')

        # Merge the Random Forest predictions with the test set for plotting
        test_set_date_dnn = test_set_date.copy()
        test_set_date_dnn = test_set_date_dnn.merge(predictions_df, how='left', on='id', suffixes=('', '_dnn'))


        st.subheader("Sales Over Time using DNN")
        st.write(predictions_df)
        # Plot the "train" dataframe in blue
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(train['date'], train['sales'], label='Train Sales', marker='o', s=10, c='blue')

        # Plot the "test_set_date" dataframe in red
        ax.scatter(test_set_date_dnn['date'], test_set_date_dnn['sales_dnn'], label='Test Sales', marker='o', s=10, c='red')

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Sales by Date')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        if st.button("Back to Algorithm Selection"):
            st.session_state.algorithm = "Algorithm Selection"

    elif st.session_state.algorithm == "XGradient Boost":
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error

        # Initialize XGBoost Regressor
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)

        # Fit the model on training data
        xgb_model.fit(X_train, y_train)

        # Predict on validation data
        y_val_pred_xgb = xgb_model.predict(X_val)

        # Calculate and display the MSE for validation set predictions
        val_mse_xgb = mean_squared_error(y_val, y_val_pred_xgb)
        st.write(f"XGBoost Validation MSE: {val_mse_xgb}")

        # Predict on the test set
        y_test_pred_xgb = xgb_model.predict(X_test)

        # Create a new dataframe with id and sales predictions from XGBoost
        xgb_predictions_df = pd.DataFrame({'id': test['id'], 'date': pd.to_datetime(test['date']).dt.date, 'sales': y_test_pred_xgb.flatten()})

        # Ensure non-negative sales predictions (adjust as needed)
        xgb_predictions_df['sales'] = xgb_predictions_df['sales'].clip(lower=0)

        # Display the XGBoost predictions dataframe
        st.write("XGBoost Sales Predictions")













        # Display the families with the most sales
        test = test.drop(columns=['sales'])
        test = test.merge(xgb_predictions_df[['id', 'sales']], on='id', how='left')

        # Filter out family columns and sales column
        family_columns = [col for col in test.columns if col.startswith('family_')]
        sales_column = 'sales'

        # Calculate the total sales for each family
        family_sales = {}
        for family in family_columns:
            family_sales[family] = test.loc[test[family] == 1, sales_column].sum()

        # Convert to DataFrame
        family_sales_df = pd.DataFrame(list(family_sales.items()), columns=['Family', 'Total Sales'])

        
        # Plotting
        st.title("Total Predicted Sales by Product Family")

        st.write(family_sales_df)

        # Create a bar plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total Sales', y='Family', data=family_sales_df, palette='viridis')

        # Display the plot
        st.pyplot(plt)



        # Display the most sales by store number
        store_sales = test.groupby('store_nbr')['sales'].sum().reset_index()

        # Sort the stores by sales
        store_sales = store_sales.sort_values(by='sales', ascending=False)

        # Streamlit app to show the bar chart
        st.title('Predicted Sales by Store')

        st.write('Predicted Sales data by store number:')
        st.dataframe(store_sales)

        # Plot the bar chart
        fig, ax = plt.subplots()
        ax.bar(store_sales['store_nbr'], store_sales['sales'])
        ax.set_xlabel('Store Number')
        ax.set_ylabel('Total Sales')
        ax.set_title('Total Sales by Store Number')

        # Display the chart in Streamlit
        st.pyplot(fig) 





        # Display Predicted Sales by City
        df_store_details = st.session_state.stores_data  # Store details with store_nbr, city, etc.

        # Filter out necessary columns for efficient lookup
        df_store_details = df_store_details[['store_nbr', 'city']]  # Only store_nbr and city columns

        # Group sales by store_nbr
        sales_by_store = test.groupby('store_nbr')['sales'].sum().reset_index()

        # Now link the two dataframes based on store_nbr (without full merge)
        # We map the city for each store_nbr in the sales_by_store DataFrame
        sales_by_store['city'] = sales_by_store['store_nbr'].map(df_store_details.set_index('store_nbr')['city'])

        # Group sales by city
        sales_by_city = sales_by_store.groupby('city')['sales'].sum().reset_index()

        # Sort by sales to find the cities with highest sales
        sales_by_city = sales_by_city.sort_values(by='sales', ascending=False)

        # Streamlit app to display the data and plot
        st.title('Sales by City')

        st.write('Total sales data grouped by city:')
        st.dataframe(sales_by_city)


        # Create a bar plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='sales', y='city', data=sales_by_city, palette='viridis')

        # Set custom labels
        ax.set_xlabel('Predicted Sales (in millions)')  # Custom x-axis label
        ax.set_ylabel('City')         # Custom y-axis label
        ax.set_title('Sales by City') # Custom title

        # Display the plot
        st.pyplot(plt)













        test_set_date=test_set_date.merge(xgb_predictions_df,how='left',on='id')

        # Merge the XGBoost predictions with the test set for plotting
        test_set_date_xgb = test_set_date.copy()
        test_set_date_xgb = test_set_date_xgb.merge(xgb_predictions_df, how='left', on='id', suffixes=('', '_xgb'))

        # Plot sales predictions from the XGBoost model
        st.subheader("Sales Over Time using XGBoost")
        st.write(xgb_predictions_df)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the "train" dataframe in blue
        ax.scatter(train['date'], train['sales'], label='Train Sales', marker='o', s=10, c='blue')

        # Plot the XGBoost predictions in orange
        ax.scatter(test_set_date_xgb['date'], test_set_date_xgb['sales_xgb'], label='XGBoost Test Sales', marker='o', s=10, c='orange')

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Sales by Date - XGBoost')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        if st.button("Back to Algorithm Selection"):
            st.session_state.algorithm = "Algorithm Selection"

def main():
    # Define the options for navigation
    options = ["Welcome", "Upload CSVs", "Algorithm Selection", "Predictions"]

    # Set the default option based on session state
    default_option = "Welcome"
    if st.session_state.algorithm == "Upload CSV":
        default_option = "Upload CSVs"
    elif st.session_state.algorithm == "Algorithm Selection":
        default_option = "Algorithm Selection"
    elif st.session_state.algorithm in ["Random Forest", "Neural Network", "XGradient Boost"]:
        default_option = "Predictions"

    # Create the sidebar with navigation options
    option = st.sidebar.radio("Select an option", options, index=options.index(default_option))

    # Render the appropriate page based on the selected option
    if option == "Welcome":
        show_welcome()
    elif option == "Upload CSVs":
        if st.session_state.algorithm == "Upload CSV":
            show_upload_csvs()
        else:
            st.write("Please complete the previous steps before uploading CSVs.")
    elif option == "Algorithm Selection":
        if st.session_state.train_data is not None:
            show_algorithm_selection()
        else:
            st.write("Please upload the CSV files first.")
    elif option == "Predictions":
        if st.session_state.algorithm in ["Random Forest", "Neural Network", "XGradient Boost"]:
            show_predictions()
        else:
            st.write("Please select an algorithm first.")

if __name__ == "__main__":
    main()





























