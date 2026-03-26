import pandas as pd
from sklearn.linear_model import LinearRegression


def run_cli():
    # Load and prepare data
    try:
        df = pd.read_csv('energy_level.csv')
    except FileNotFoundError:
        print("File energy_level.csv not found. Make sure it's in the same folder.")
        return

    if not all(col in df.columns for col in ['sleep_hours', 'break_time', 'energy_level']):
        print("Required columns missing. CSV needs sleep_hours, break_time, and energy_level.")
        return

    x = df[['sleep_hours', 'break_time']]
    y = df['energy_level']  # Use 1D series

    # Train model
    model = LinearRegression()
    model.fit(x, y)

    # Get user input
    try:
        user_input = float(input("Enter your sleep hours: "))
        user_input2 = float(input("Enter your break time in hours: "))
    except ValueError:
        print("Invalid input. Please enter numbers.")
        return

    # Make prediction
    input_data = pd.DataFrame({
        'sleep_hours': [user_input],
        'break_time': [user_input2]
    })
    y_pred = model.predict(input_data)

    # Display result
    print(f"Predicted energy level: {y_pred[0]:.2f}")
    print("predicted energy level:", y_pred)


def run_streamlit_app():
    try:
        import streamlit as st
    except ImportError:
        raise ImportError('Streamlit is not installed. Run `pip install streamlit`.')

    st.title('Energy Level Prediction')
    st.write('Predict your energy level using sleep and break time.')

    sleep_hours = st.slider('Sleep hours', 0.0, 16.0, 7.0, step=0.25)
    break_time = st.slider('Break time (hours)', 0.0, 6.0, 1.0, step=0.1)

    model_option = st.selectbox('Regression model', ['LinearRegression'], index=0)

    if st.button('Predict'):
        try:
            df = pd.read_csv('energy_level.csv')
        except FileNotFoundError:
            st.error('File energy_level.csv not found. Make sure it is in the same folder.')
            return

        if not all(col in df.columns for col in ['sleep_hours', 'break_time', 'energy_level']):
            st.error('Required columns missing. CSV needs sleep_hours, break_time, and energy_level.')
            return

        x = df[['sleep_hours', 'break_time']]
        y = df['energy_level']

        model = LinearRegression()
        model.fit(x, y)

        input_data = pd.DataFrame({'sleep_hours': [sleep_hours], 'break_time': [break_time]})
        y_pred = model.predict(input_data)

        st.success(f'Predicted energy level: {y_pred[0]:.2f}')
        st.write('Predicted energy level array:', y_pred)


if __name__ == '__main__':
    try:
        import streamlit
        run_streamlit_app()
    except Exception:
        run_cli()
