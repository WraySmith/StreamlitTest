import streamlit as st
import covid_app_func as func
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Load Data
def load_data():
    data_path = './data/covid/covid19_20200725_mod.csv'
    column_interest = 'new_count_update'
    n_mean = 3
    df = func.process_data(data_path, column_interest)
    df_out, plot_out = func.mean_data(df, n_mean)
    df_out, lam_out = func.add_series(df_out, columns_analyze = 'daily_case')
    return df_out, lam_out, plot_out

def main():
    st.title('BC COVID Cases Explorer')
    # Add a selectbox to the sidebar:
    select_menu = st.sidebar.selectbox(
        'Choose Page',
        ('Explore Data', 'Explore ARIMA', 'ARIMA Modelling')
    )
    covid_data, lam_print, mean_plot = load_data()

    if select_menu == "Explore Data":
        st.pyplot(mean_plot)
        st.text('Lambda of Box-Cox Transform: ' + str(lam_print))
        st.write(covid_data.head())
        col_explore = ['daily_case',  'daily_case_diff', 'daily_case_bc', \
            'daily_case_bc_diff', 'daily_case_2wknormal', 'daily_case_2wknormal_diff']
        explore_data, explore_plot = func.explore_series(covid_data, col_explore)
        st.pyplot(explore_plot)
        st.write(explore_data)

    if select_menu == "Explore ARIMA":

        select_splits = st.sidebar.slider(
            'Select Number of Training Splits',
            2, 5, key = 'slider1' 
        )
        select_p = st.sidebar.slider(
            'Select P parameter range',
            0, 20
        )
        select_d = st.sidebar.slider(
            'Select D parameter range',
            0, 2
        )
        select_q = st.sidebar.slider(
            'Select Q parameter range',
            0, 20
        )
        
        train_check1 = st.checkbox('Show Train Test Splits', value=False, key='train_check1')
        if train_check1:
            split_fig = func.splitter(covid_data, 'daily_case', select_splits) 
            st.pyplot(split_fig)

        p_values = range(0, select_p+1)
        d_values = range(0, select_d+1)
        q_values = range(0, select_q+1)

        if st.sidebar.button('Run', key='run_explore'):
            warnings.filterwarnings("ignore")
            counter = 0
            for train_index, test_index in TimeSeriesSplit(select_splits).split(covid_data['daily_case']):
                counter += 1
                train = covid_data['daily_case'][train_index]
                test = covid_data['daily_case'][test_index]
                temp_results, temp_best = func.grid_ARIMA_models(train, test, p_values, d_values, q_values)
                temp_plot = func.pdq_heat(temp_results)
                text_best = ''.join(["Test Set ", str(counter), ": ", temp_best])
                st.text(text_best)
                st.pyplot(temp_plot)

    if select_menu == 'ARIMA Modelling':
        ARIMA_run = st.sidebar.button('Run', key='run_explore')
        
        select_splits = st.sidebar.slider(
            'Select Number of Training Splits',
            2, 5, key = 'slider1'
        )
        select_p_model = st.sidebar.slider(
            'Select P parameter',
            0, 20
        )
        select_d_model = st.sidebar.slider(
            'Select D parameter',
            0, 2
        )
        select_q_model = st.sidebar.slider(
            'Select Q parameter',
            0, 20
        )
        select_ARIMA = st.sidebar.selectbox(
            'Choose Type',
            ('Single-Step', 'Multi-Step')
        )

        train_check2 = st.checkbox('Show Train Test Splits', value=False, key='train_check2')
        if train_check2:
            split_fig = func.splitter(covid_data, 'daily_case', select_splits) 
            st.pyplot(split_fig)

        covid_data_results = covid_data[['days_elapse', 'daily_case']].copy()
        # evaluate models
        if select_ARIMA == 'Single-Step':
            if ARIMA_run:
                covid_data_results, rmse_results = func.evaluate_models(splits = TimeSeriesSplit(select_splits), \
                                                                    df = covid_data_results, data_col = 'daily_case', \
                                                                    arima_order = (select_p_model,select_d_model,select_q_model))
                st.write(covid_data_results.head())
                st.write(rmse_results)
                labels_plot = [("_" + str(i+1)) for i in range(select_splits)]
                results_plot = func.compare_plot(covid_data_results, index_col = 'days_elapse', \
                                        labels = labels_plot, original = 'daily_case')
                st.pyplot(results_plot)
        else:
            select_steps = st.sidebar.slider(
            'Select Forecast Steps',
            1, 10, 2
            )
            if ARIMA_run:
                covid_data_results, rmse_results = func.multistep_forecast(splits = TimeSeriesSplit(select_splits), 
                                                                        df = covid_data_results, 
                                                                        data_col = 'daily_case',
                                                                        arima_order = (select_p_model,select_d_model,select_q_model), 
                                                                        forecast_steps = select_steps, 
                                                                        confidence = False)
                st.write(covid_data_results.head())
                st.write(rmse_results)
                labels_plot = [("_" + str(i+1)) for i in range(select_splits)]
                results_plot = func.compare_plot(covid_data_results, index_col = 'days_elapse', \
                                        labels = labels_plot, original = 'daily_case')
                st.pyplot(results_plot)




if __name__ == "__main__":
    main()








