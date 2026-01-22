import pandas as pd
import jieba
import os
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI


def text_calssify_using_ml(text:str):
    dataset=pd.read_csv(r"C:\Users\Admin\PycharmProjects\Homework\Week01\dataset.csv",sep="\t",header=None,nrows=10000)
    print(dataset[0].value_counts())
    input_sententce=dataset[0].apply(lambda x: " ".join(jieba.cut(x)))
    print (input_sententce)
    vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
    vector.fit(input_sententce.values)  # 统计词表
    input_feature = vector.transform(input_sententce.values)  # 进行转换 100 * 词表大小

    model = KNeighborsClassifier()
    model.fit(input_feature, dataset[1].values)
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_lmd(text:str):
    client = OpenAI(
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 账号绑定，用来计费的

        # 大模型厂商的地址，阿里云
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[{"role": "user", "content": f"""帮我进行文本分类：{text}
        输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""}]
    ) # 用户的提问
    return completion.choices[0].message.content

if __name__ == "__main__":
    #dataset=pd.read_csv(r"C:\Users\Admin\PycharmProjects\Homework\Week01\dataset.csv",sep="\t",header=None)
    print("机器学习预测：",text_calssify_using_ml('我下个月去北京天安门'))
    print("大模型预测：",text_calssify_using_lmd('我下个月去北京'))
