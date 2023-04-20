import streamlit as st

# Maybe create a streamlit dashboard to allow people to interact with the models?

# Issues:
#     There are 120 entries required to make one prediction - is that too much to bother with? 


class homepage:
    def __init__(self) -> None:
        st.header("Credit Application Prediction App")
        st.markdown(
            "This dashboard will allow you to enter all required information and then receive a prediction on whether you'd be accepted or not"
        )
        self.get_inputs()

    def get_inputs(self):
        age = st.text_input("Age")
        st.button("Make predictions", on_click=self.make_predictions)

    def make_predictions(self): 
        pass

    def present_predictions(self):
        pass


homepage()
