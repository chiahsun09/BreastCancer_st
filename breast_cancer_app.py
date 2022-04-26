
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

#
cancer  = load_breast_cancer()


# separate the data into features and target
features = pd.DataFrame(
    cancer.data, columns=cancer.feature_names
)
#cancer.target_names=np.array(['benign','malignant'], dtype='<U9')
#cancer.target=np.invert(cancer.target)+2



target = pd.Series(cancer.target)


#只取之前分析的最主要五個特徵來跑模型
features=features[['mean concave points', 'radius error', 'worst radius', 'worst texture', 'worst concave points']]


# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42,stratify=target
)

#常態分佈化,使離群值影響降低
x_train=(x_train - x_train.mean()) / (x_train.std())
x_test=(x_test - x_test.mean()) / (x_test.std())


class StreamlitApp:
    st.image("https://storage.googleapis.com/kaggle-datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-cover.jpg")

    def __init__(self):
        self.model = SGDClassifier(
                alpha=0.01,
                average=True,
                l1_ratio=0.0,
                loss= 'log',
                penalty='elasticnet',
                max_iter=5)


    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        st.sidebar.markdown(
            '<p class="header-style">Breast_cancer Classification</p>',
            unsafe_allow_html=True
        )

        mean_concave_points = st.sidebar.slider(
            f"Scroll{cols[0]}", 
            0.00,   #features[cols[0]].min()
            features[cols[0]].max(), 
            features[cols[0]].median()
        )  

        radius_error = st.sidebar.slider(
            f"Scroll{cols[1]}", 
            0.00, # 最小值features[cols[1]].min()
            features[cols[1]].max(), 
            features[cols[1]].median()
        )  

        worst_radius = st.sidebar.slider(
            f"Scroll{cols[2]}", 
            0.00, #最小值features[cols[2]].min()
            features[cols[2]].max(), 
            features[cols[2]].median()
        )  

        worst_texture = st.sidebar.slider(
            f"Scroll{cols[3]}", 
            0.00, #最小值features[cols[3]].min()
            features[cols[3]].max(), 
            features[cols[3]].median()
        )  

        worst_concave_points = st.sidebar.slider(
            f"Scroll{cols[4]}", 
            0.00, #最小值features[cols[4]].min()
            features[cols[4]].max(), 
            features[cols[4]].median()
        )  


        #把輸入值作常態分佈處理
        input_list=[mean_concave_points,radius_error,worst_radius,worst_texture,worst_concave_points]
        for i, element in enumerate(features):
            input_list[i]=(input_list[i] - features[element].mean()) / (features[element].std())
        
        values = input_list
        return values
        
        '''
        同上程式,白話文解說
        mean_concave_points=(mean_concave_points - features[cols[0]].mean()) / (features[cols[0]].std())
        radius_error=(radius_error - features[cols[1]].mean()) / (features[cols[1]].std())
        worst_radius=(worst_radius - features[cols[2]].mean()) / (features[cols[2]].std())
        worst_texture=(worst_texture - features[cols[3]].mean()) / (features[cols[3]].std())
        worst_concave_points=(worst_concave_points - features[cols[4]].mean()) / (features[cols[4]].std())
        values = [mean_concave_points, radius_error, worst_radius, worst_texture, worst_concave_points]
        return values
        '''
        

    def plot_pie_chart(self, probabilities):
        colors = ['#FD3D5A', '#A9DABE']
        fig = go.Figure(
            data=[go.Pie(
                    labels=['malignant','benign'],
                    values=probabilities[0]
                    
            )]
        )
        fig = fig.update_traces(
            hoverinfo='label+percent',
            textinfo='value',
            textfont_size=15,
            marker=dict(colors=colors, line=dict(color='#000000', width=1))
        )
        return fig

    def construct_app(self):

        self.train_data()
        values = self.construct_sidebar()

        values_to_predict = np.array(values).reshape(1, -1)

        prediction = self.model.predict(values_to_predict)
        prediction_str = cancer.target_names[prediction[0]]
        probabilities = np.round(self.model.predict_proba(values_to_predict),2)

        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Breast_cancer Predictions </p>',
            unsafe_allow_html=True
        )

        column_1, column_2 = st.columns(2)
        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0][prediction[0]]}")

        fig = self.plot_pie_chart(probabilities)
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

        return self


sa = StreamlitApp()
sa.construct_app()
