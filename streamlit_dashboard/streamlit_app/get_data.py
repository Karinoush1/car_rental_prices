import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# load the data
df_price = pd.read_csv('/Users/karinapopova/Desktop/JEDHA/M11_CERTIF/C_depl/get_around_pricing_project.csv')
df_delay = pd.read_excel('/Users/karinapopova/Desktop/JEDHA/M11_CERTIF/C_depl/get_around_delay_analysis.xlsx')
# create a sub-dataset where we have information about previous rentals.
df_previous_rental_known = df_delay[df_delay['previous_ended_rental_id'].notna()]

st.title('Getaround Pricing Project Insights')

def display_general_statistics():
    # Define the options
    # Define the options
    checkin_type_options = ['all-types', 'connect', 'mobile']

    # Create a select box
    selected_checkin_type = st.selectbox(
        "Select Check-In Type:",
        checkin_type_options
        )
    # Filter the dataframe based on the selection
    if selected_checkin_type != 'all-types':
        df_to_use = df_previous_rental_known[df_previous_rental_known['checkin_type'] == selected_checkin_type]
    else:
        df_to_use = df_previous_rental_known

    # Late check-outs
    st.header('Late check-outs')
    num_late = df_to_use[df_to_use['delay_at_checkout_in_minutes'] >= 0].shape[0]
    percentage_late = (num_late / len(df_to_use)) * 100
    # Calculate the median and quarter delay
    filtered_df = df_to_use[df_to_use['delay_at_checkout_in_minutes'] >= 0]
    median_delay = filtered_df['delay_at_checkout_in_minutes'].quantile(0.5)
    quater_delay = filtered_df['delay_at_checkout_in_minutes'].quantile(0.25)

    # Display information
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between;'>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1; background-color: #3D82CD;'>
            <h1>{num_late}</h1>
            <p> of {len(df_to_use)} rentals who did {selected_checkin_type} check-in are late</p>
        </div>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1; background-color: #3D82CD;'>
            <h1>{round(percentage_late, 2)}%</h1>
            <p> of rentals who did {selected_checkin_type} check-in are late</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='display: flex; justify-content: space-between;'>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1;background-color: #FC8D43'>
            <h1>{median_delay} min</h1>
            <p>50% of delays are less than this value.</p>
        </div>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1;background-color: #FC8D43'>
            <h1>{quater_delay} min</h1>
            <p>25% of delays are less than this value.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Late checkouts affecting next rentals
    st.header('Late check-outs affecting next rentals')
    delay_bigger_than_delta = df_to_use[df_to_use['delay_at_checkout_in_minutes'] > df_to_use['time_delta_with_previous_rental_in_minutes']]
    percentage_delay_bigger_than_delta = (len(delay_bigger_than_delta) / len(df_to_use)) * 100
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between;'>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1; '>
            <h1>{len(delay_bigger_than_delta)}</h1>
            <p>Number of rentals affected by a delay</p>
        </div>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1; '>
            <h1>{round(percentage_delay_bigger_than_delta, 2)}%</h1>
            <p>Percentage of these rentals</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # delay which realy afected next rental (late-delta)
    delay_bigger_than_delta['time_diff'] = delay_bigger_than_delta['delay_at_checkout_in_minutes'] - delay_bigger_than_delta['time_delta_with_previous_rental_in_minutes']
    delta_to_study= 210
    filtered_time_diff = delay_bigger_than_delta[delay_bigger_than_delta['time_diff'] <= delta_to_study]['time_diff']
    plt.hist(filtered_time_diff, bins=14)
    plt.xlim(0, 210)
    x_ticks = [0,15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210]
    plt.xticks(x_ticks)
    plt.xlabel('Delay affecting next rental(minutes)')
    plt.ylabel('Number of rentals')
    plt.title('Distribution of Delays affecting next rental')
    st.pyplot(plt)


def display_decision_making():
    #st.title('Decision Making')
    threshold = st.slider('Select the minimum delay threshold (minutes):', 1, 120, 30)


    #### How many problematic cases will it solve depending on the chosen threshold and scope?
    st.header('How many problematic cases this threshold will solve?')
    total_values = len(df_delay)
    num_affected = df_delay[(df_delay['delay_at_checkout_in_minutes'] >= 0) & (df_delay['delay_at_checkout_in_minutes'] <= threshold)]
    #st.write(f"Number of cases that this threshold will solve: {len(num_affected)}" )
    percentage_affected = (len(num_affected) /  len(df_delay)) * 100
    #st.write(f"Percentage of cases that this threshold will solve: {percentage_affected}" )
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between;'>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1; background-color: #FC8D43;'>
            <h1>{len(num_affected)}</h1>
            <p>Number of cases that this threshold will solve</p>
        </div>
        <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; flex: 1; background-color: #FC8D43;'>
            <h1>{round(percentage_affected, 2)}%</h1>
            <p>Percentage of cases that this threshold will solve</p>
        </div>
    </div>
""", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")

    # What share will be affected?
    st.header('Which share of our owner’s revenue would potentially be affected by the feature?')
    # Ask the user for the rental duration
    x = st.number_input("How long is the rent (in hours)?", min_value=1, step=1)

    min_day = 1440
    min_x = min_day/(24/x)
    #treshold= 30
    df_price['rental_price_x']= df_price['rental_price_per_day']/(24/x)
    df_price['delay_cost_x'] = (df_price['rental_price_x'] /min_x)  * threshold
    df_price['delay_percentage_x'] = 100 * df_price['delay_cost_x']/df_price['rental_price_x']
    delay_cost_percentage = df_price['delay_percentage_x'].iloc[0]
    st.markdown(f"""
    <div style='text-align: center; border:1px solid black; padding:20px; margin:5px; '>
        <h2>{round(delay_cost_percentage, 2)}%</h2>
        <p>Percentage of revenue affected by this threshold</p>
    </div>
    """, unsafe_allow_html=True)
    exemple_price = 120
    st.write(f" For exemple, for a car whose  rental price per day is {exemple_price}, rented for {x} hours, this threshold's cost will be {round(exemple_price /1440 * threshold, 2)}. This cost represents {round((exemple_price / 1440 * threshold)*100/exemple_price*(24/x), 2)} % of the retal income. ")
    st.dataframe(df_price)


    # Then, create a new function for displaying the raw data
def display_raw_data():
    # Assuming df1 and df2 are your dataframes
    st.header('Raw Data')
    dataset = st.selectbox(
        "Select Dataset",
        ["Pricing by car", "Delay data"]
    )

    if dataset == "Pricing by car":
        st.subheader("Pricing by car")
        st.write(df_price)

    else:
        st.subheader("Delay data")
        st.write(df_delay)
        st.write("")
        st.write("")
        st.write("")
        st.subheader('Filtered "Delay data" where we have information about previous rentals')
        st.write("The information about previous rentals is essential in uderstanding the impact of delays on further rentals. That is the reason why this was the dataset used in 'General Statistics' and 'Descision Making. ")
        st.write(df_previous_rental_known)







# Chose the page you want to explore
page = st.sidebar.selectbox(
    "Choose a page",
    ["General Statistics", "Decision Making", "View Raw Data"]
)

# Display the selected page
if page == "General Statistics":
    display_general_statistics()
elif page == "Decision Making":
    display_decision_making()
elif page == "View Raw Data":
    display_raw_data()
