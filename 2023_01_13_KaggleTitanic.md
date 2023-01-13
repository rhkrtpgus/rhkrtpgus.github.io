{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "#**활동내용**"
      ],
      "metadata": {
        "id": "6dN8__USTDlD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###**케글 대회 소개**"
      ],
      "metadata": {
        "id": "hXhtM_8vUBQt"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "타이타닉 생존자 분석\n",
        "[주소] - https://www.kaggle.com/competitions/titanic\n",
        "\n",
        "타이타닉 생존자 예측\n",
        "\n",
        "데이터: 타이타닉 탑증자의 여러 정보(예:나이, 성별, 탑승장소 등등)\n",
        "\n",
        "목적: 미리 제공된 약 800명의 데이터를 가지고 모델을 학습시켜 테스트 인원들의 생존여부 분석. 생존여부를 정확하게 예측한 정도 즉 정확도를 점차 높이는것이 궁극적인 목적이다."
      ],
      "metadata": {
        "id": "byKM8HXSv7x4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###**진행평가요소**\n",
        "\n",
        ">1. 데이터 전처리 방법\n",
        ">2. 모델 선정방법 및 이유\n",
        "\n",
        "###**피드백**\n",
        "개인피드백\n",
        "> 결측치와 각 열중에서 필요한것과 필요하지 않은 것들을 구분하여 상관관계 분석하고 어떤 것을 쓰고 쓰지 않을지 결정하여야한다.\n",
        "\n",
        "###**깃허브**\n",
        "> 대다수의 스터디 인원들이 깃허브 사용경험이 적어 이번 모각소 스터디를 계기로 깃허브 레포짓 및 블로그 작성하는 연습을 시행하였습니다. 이후 사용에 익숙해지면 깃허브 팀 레포짓을 만들어 팀단위로 대회에 나가는 증 해볼 계획입니다.\n",
        "\n",
        "#**활동사진**\n",
        "2시 활동사진\n",
        "\n",
        "![2시 활동사진](/images/2023-01-13-MGS1/2시 활동사진-1673591971027-1.png)\n",
        "\n",
        "3시 활동사진\n",
        "\n",
        "![3시 활동사진](/images/2023-01-13-MGS1/3시 활동사진.png)\n",
        "\n",
        "4시 활동사진\n",
        "\n",
        "![4시 활동사진](/images/2023-01-13-MGS1/4시 활동사진.png)\n",
        "\n",
        "5시 종료사진\n",
        "\n",
        "#**개인활동내역**\n",
        "\n",
        "###**개인 활동 블로그 url**\n",
        "*전장훈: https://jhwannabe.tistory.com/25\n",
        "\n",
        "*신재현: https://hectorsin.github.io/categories/#mgs\n",
        "\n",
        "*곽세현: https://rhkrtpgus.github.io\n",
        "\n",
        "*강성현: https://seong-hyeon-2.github.io\n",
        "\n",
        "*김수진: https://sujin7822.github.io/\n",
        "\n",
        "###**개인 깃 TIL**\n",
        "*전장훈: https://github.com/JHWannabe/TIL\n",
        "\n",
        "*신재현: https://github.com/HectorSin/TIL\n",
        "\n",
        "*곽세현: https://github.com/rhkrtpgus/TIL\n",
        "\n",
        "*강성현: https://github.com/seong-hyeon-2/TIL\n",
        "\n",
        "*김수진: https://github.com/sujin7822/TIL"
      ],
      "metadata": {
        "id": "9X-HaGLAUpqW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 분석에 필요한 패키지 임포트 및 데이터 불러오기\n",
        "# 데이터 분석 및 정리 패키지\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import random as rnd\n",
        "\n",
        "# 데이터 시각화 패키지\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# 머신러닝용 패키지\n"
      ],
      "metadata": {
        "id": "GIhCstuewNM9"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###* pandas 사용 이유 \n",
        ">##### - 데이터 분석에 특화된 자료구조를 제공하는 라이브러리 인데 스트레드시트 구조로 데이터를 다룰 수 있어서 가장 직관적이다.\n",
        "\n",
        "###* numpy 사용 이유\n",
        ">##### - 데이터의 대부분은 숫자 배열로 볼 수 있다. numpy는 비교적 빠른 연산을 지원하고 메모리를 효율적으로 사용한다\n",
        "\n",
        "###* matplotlib 사용 이유 (vs. seaborn)\n",
        ">##### - 막대그래프\n",
        ">##### - 선 그래프\n",
        ">##### - 산점도\n",
        ">##### - ETC 등의 시각화를 진행하기에 용이하다\n"
      ],
      "metadata": {
        "id": "nKLEQzdCxk3Y"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **1. 데이터 불러오기**"
      ],
      "metadata": {
        "id": "fXszSHkS0V1N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WrSOMM3M1mDX",
        "outputId": "cf92737f-83b6-439d-f5eb-24bbd739b2bc"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df = pd.read_csv('/content/drive/MyDrive/All-in/train.csv')\n",
        "test_df = pd.read_csv('/content/drive/MyDrive/All-in/test.csv') \n",
        "\n",
        "# 데이터 요약\n",
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        },
        "id": "xU-zD1Wq0anO",
        "outputId": "d4913a8e-d490-47c0-e2fe-c3ba7c2e457f"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e72ce78e-107e-44c4-bd66-87753d1680eb\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e72ce78e-107e-44c4-bd66-87753d1680eb')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e72ce78e-107e-44c4-bd66-87753d1680eb button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e72ce78e-107e-44c4-bd66-87753d1680eb');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 마지막 데이터\n",
        "train_df.tail()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 285
        },
        "id": "-b5nMxcF9ew0",
        "outputId": "152231a4-165a-4a5a-dc90-3fc6cbf9edfb"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     PassengerId  Survived  Pclass                                      Name  \\\n",
              "886          887         0       2                     Montvila, Rev. Juozas   \n",
              "887          888         1       1              Graham, Miss. Margaret Edith   \n",
              "888          889         0       3  Johnston, Miss. Catherine Helen \"Carrie\"   \n",
              "889          890         1       1                     Behr, Mr. Karl Howell   \n",
              "890          891         0       3                       Dooley, Mr. Patrick   \n",
              "\n",
              "        Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  \n",
              "886    male  27.0      0      0      211536  13.00   NaN        S  \n",
              "887  female  19.0      0      0      112053  30.00   B42        S  \n",
              "888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  \n",
              "889    male  26.0      0      0      111369  30.00  C148        C  \n",
              "890    male  32.0      0      0      370376   7.75   NaN        Q  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f3d7b649-a959-4bfd-8a89-1b2aabe4b100\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>886</th>\n",
              "      <td>887</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>Montvila, Rev. Juozas</td>\n",
              "      <td>male</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>211536</td>\n",
              "      <td>13.00</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>887</th>\n",
              "      <td>888</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Graham, Miss. Margaret Edith</td>\n",
              "      <td>female</td>\n",
              "      <td>19.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>112053</td>\n",
              "      <td>30.00</td>\n",
              "      <td>B42</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>888</th>\n",
              "      <td>889</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
              "      <td>female</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>W./C. 6607</td>\n",
              "      <td>23.45</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>889</th>\n",
              "      <td>890</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Behr, Mr. Karl Howell</td>\n",
              "      <td>male</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>111369</td>\n",
              "      <td>30.00</td>\n",
              "      <td>C148</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>890</th>\n",
              "      <td>891</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Dooley, Mr. Patrick</td>\n",
              "      <td>male</td>\n",
              "      <td>32.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>370376</td>\n",
              "      <td>7.75</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f3d7b649-a959-4bfd-8a89-1b2aabe4b100')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f3d7b649-a959-4bfd-8a89-1b2aabe4b100 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f3d7b649-a959-4bfd-8a89-1b2aabe4b100');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **2. 데이터 살펴보기**\n",
        "##### * 데이터의 형태와 크기를 알아보고 결측치를 파악하여 어떠한 형태로 가공할 것인지 방향을 정한다."
      ],
      "metadata": {
        "id": "15Xr2Dfo3YV9"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        ">### **2.1 데이터 딕셔너리**\n",
        ">> ##### survival = 생존여부 (0=no, 1=yes)\n",
        ">> ##### pclass = 사회, 경제적 지위 (1=1st, 2=2st, 3=3st)\n",
        ">> ##### sex = 성별\n",
        ">> ##### age = 나이\n",
        ">> ##### sibsp = 타이타닉호에 탑승한 형제, 자매 수\n",
        ">> ##### parch = 타이타닉호에 탑승한 부모, 자녀 수\n",
        ">> ##### ticket =  티켓 번호\n",
        ">> ##### fare = 탑승 요금\n",
        ">> ##### cabin = 탑승 요금\n",
        ">> ##### embarked = 탑승 지역(항구 위치)\n",
        "\n",
        ">### **2.2 결측치 파악**\n",
        ">>##### 데이터의 형태 알아보기기"
      ],
      "metadata": {
        "id": "n3xsDzrj6Rv1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S8B_mLQO7iRh",
        "outputId": "49f2eea6-800a-46d1-8878-22769ac4e718"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(891, 12)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_df.shape \n",
        "# 테스트 데이터는 훈련 데이터로 학습 시킨 모델을 통해 라벨링 해야하므로 survived 열이 빠져서 11개 인것"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BlD7T8Th48ul",
        "outputId": "721e0589-c369-4a14-c009-78e7a042871e"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(418, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# train 데이터의 정보 \n",
        "train_df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s32Kv9398ezZ",
        "outputId": "c9f51893-ad1c-4ba6-82c3-5c2f4bc306ff"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# test 데이터의 정보\n",
        "test_df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qFtzgq-n8Hxe",
        "outputId": "11dc8b6b-28cc-4f36-e7ed-b52b94d958d0"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 418 entries, 0 to 417\n",
            "Data columns (total 11 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  418 non-null    int64  \n",
            " 1   Pclass       418 non-null    int64  \n",
            " 2   Name         418 non-null    object \n",
            " 3   Sex          418 non-null    object \n",
            " 4   Age          332 non-null    float64\n",
            " 5   SibSp        418 non-null    int64  \n",
            " 6   Parch        418 non-null    int64  \n",
            " 7   Ticket       418 non-null    object \n",
            " 8   Fare         417 non-null    float64\n",
            " 9   Cabin        91 non-null     object \n",
            " 10  Embarked     418 non-null    object \n",
            "dtypes: float64(2), int64(4), object(5)\n",
            "memory usage: 36.0+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# train 데이터셋 설명\n",
        "train_df.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "RyS0H0vL980r",
        "outputId": "719fd276-4806-4f80-acd3-1dadf7c63970"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
              "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
              "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
              "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
              "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
              "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
              "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
              "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
              "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
              "\n",
              "            Parch        Fare  \n",
              "count  891.000000  891.000000  \n",
              "mean     0.381594   32.204208  \n",
              "std      0.806057   49.693429  \n",
              "min      0.000000    0.000000  \n",
              "25%      0.000000    7.910400  \n",
              "50%      0.000000   14.454200  \n",
              "75%      0.000000   31.000000  \n",
              "max      6.000000  512.329200  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0c176494-2e01-4727-9ed0-1ad132e82f70\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>714.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.383838</td>\n",
              "      <td>2.308642</td>\n",
              "      <td>29.699118</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>32.204208</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>257.353842</td>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>14.526497</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>49.693429</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.420000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>223.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>20.125000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>7.910400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.454200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>668.500000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>38.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>31.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>512.329200</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0c176494-2e01-4727-9ed0-1ad132e82f70')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-0c176494-2e01-4727-9ed0-1ad132e82f70 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-0c176494-2e01-4727-9ed0-1ad132e82f70');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# null의 갯수 세어보기\n",
        "train_df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sGfbOfz-8cno",
        "outputId": "4e82ac4d-6ffd-43a0-c2a5-e220dedb55d5"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId      0\n",
              "Survived         0\n",
              "Pclass           0\n",
              "Name             0\n",
              "Sex              0\n",
              "Age            177\n",
              "SibSp            0\n",
              "Parch            0\n",
              "Ticket           0\n",
              "Fare             0\n",
              "Cabin          687\n",
              "Embarked         2\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xLyfuUOm9GGS",
        "outputId": "b6ac3517-56a1-4c1a-ac85-896b41f16bf7"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId      0\n",
              "Pclass           0\n",
              "Name             0\n",
              "Sex              0\n",
              "Age             86\n",
              "SibSp            0\n",
              "Parch            0\n",
              "Ticket           0\n",
              "Fare             1\n",
              "Cabin          327\n",
              "Embarked         0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">>#####null 값이 있으면 모델이 안돌아간다. 정확한 갯수를 세는 이유는 결측치를 없애기 위함이고, 결측치가 너무 많으면 어떤것으로 대체해야하는지 알기 위해서\n",
        "\n",
        ">>#####반 이상이 결측치다 -> 평균값으로 대체하는거보다 결측치로 대체\n",
        ">>#####소규모 -> 평균값으로 대체하거나 데이터에 특성에 따라서 처리한다"
      ],
      "metadata": {
        "id": "q195xgMfGGse"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        ">### **2.3 데이터 시각화**\n",
        ">> 열들을 시각화 하여 대략적인 분포를 파악하고, 생존과의 상관관계를 유추한다. 명목형, 이산형 데이터에 대해 막대 그래프를 그려준다. "
      ],
      "metadata": {
        "id": "VoJF95b7-O2u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def bar_chart(feature):\n",
        "  # 각 column(=feature)에서 생존자의 수 count\n",
        "  survived = train_df[train_df['Survived']==1][feature].value_counts()\n",
        "\n",
        "  # 각 column(=feature)에서 사망자의 수 count\n",
        "  dead = train_df[train_df['Survived']==0][feature].value_counts()\n",
        "\n",
        "  # 생존자의 수, 사망자의 수를 하나로 묶기\n",
        "  df= pd.DataFrame([survived,dead])\n",
        "\n",
        "  # 묶은 dataframe의 인덱스명(행이름) 지정\n",
        "  df.index=['Survived','Dead']\n",
        "\n",
        "  # plot 그리기\n",
        "  df.plot(kind='bar', stacked=True, figsize=(10,5))"
      ],
      "metadata": {
        "id": "Qc1pwiCI_0hw"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 성별 차트\n",
        "bar_chart('Sex') # 남성이 여성에 비해 더 많이 사망한 것을 알 수 있다"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "id": "UHG0OAZc9LkB",
        "outputId": "4e1602a2-5c60-4463-9604-c3e79122d4de"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlYAAAFRCAYAAAC2SOM6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAXw0lEQVR4nO3dfbSdVX0n8O8vgKS2CBUjiya0yYwIyABGI4K0HYGxiNXC0mp1saagrMlyadf0xdFRa6aMjq3OYnQUxxcsXUAX1peikmmdKYLBlzoaghFqCw4RI4SxJUZJITSa6J4/7gNewo33kuzknHP7+ax113n2fvZ5nt/JH5cv+9ln32qtBQCAvbdg1AUAAMwXghUAQCeCFQBAJ4IVAEAnghUAQCeCFQBAJweOuoAkecITntCWLl066jIAAGZ10003fae1tmimc2MRrJYuXZp169aNugwAgFlV1bd2d86jQACATgQrAIBOBCsAgE7GYo3VTHbs2JFNmzZl+/btoy5lrCxcuDBLlizJQQcdNOpSAIBdjG2w2rRpUw455JAsXbo0VTXqcsZCay1btmzJpk2bsmzZslGXAwDsYmwfBW7fvj2HH364UDVNVeXwww83iwcAY2psg1USoWoG/k0AYHyNdbAatXe/+9057rjjct555+2T61900UW5+OKL98m1AYD9b2zXWO1q6ev/suv1Nr7tV2cd8973vjfXXXddlixZ0vXeAMD8ZMZqN175ylfmjjvuyNlnn523vvWtecUrXpGTTz45y5cvzzXXXJMkufzyy3PuuefmOc95TpYuXZr3vOc9ecc73pHly5fnlFNOyXe/+90kyQc/+ME84xnPyEknnZQXvehFeeCBBx5xv2984xt57nOfm6c//en5pV/6pdx222379fMCAHtPsNqN97///fm5n/u5rFmzJtu2bcsZZ5yRtWvXZs2aNXnta1+bbdu2JUm+9rWv5eMf/3huvPHG/P7v/34e+9jHZv369Tn11FNz5ZVXJkle+MIX5sYbb8zNN9+c4447Lpdddtkj7rdy5cpccskluemmm3LxxRfnVa961X79vADA3puYR4GjdO2112b16tUPrYfavn177rzzziTJ6aefnkMOOSSHHHJIDj300LzgBS9Ikpxwwgm55ZZbkkyFrze96U259957c//99+ess8562PXvv//+fPGLX8yLX/zih/q+//3v74+PBvxzdNGho66ASXHR1lFXMHEEqzloreXqq6/OMccc87D+L3/5yzn44IMfai9YsOCh9oIFC7Jz584kyQUXXJBPfvKTOemkk3L55ZfnhhtueNh1fvSjH+Wwww7LV7/61X37QQCAfcqjwDk466yzcskll6S1liRZv379o3r/fffdlyOPPDI7duzIVVdd9Yjzj3vc47Js2bJ87GMfSzIV5G6++ea9LxwA2K8EqzlYtWpVduzYkRNPPDHHH398Vq1a9aje/5a3vCXPfOYzc9ppp+XYY4+dccxVV12Vyy67LCeddFKOP/74hxbIAwCTox6chRmlFStWtHXr1j2s79Zbb81xxx03oorGm38bYK9YY8VcWWM1o6q6qbW2YqZzZqwAADoRrAAAOhGsAAA6EawAADoRrAAAOhGsAAA6Eaz2kRtuuCHPf/7zR10GALAfTc6ftOm974q9OQCAzsxY/QQbN27MsccemwsuuCBPfvKTc9555+W6667LaaedlqOPPjpr167N2rVrc+qpp2b58uV51rOela9//euPuM62bdvyile8IieffHKWL19uV3UAmKcEq1ls2LAhr3nNa3Lbbbfltttuy4c+9KF84QtfyMUXX5w//MM/zLHHHpvPf/7zWb9+fd785jfnjW984yOu8da3vjVnnHFG1q5dmzVr1uS1r31ttm3bNoJPAwDsS5PzKHBEli1blhNOOCFJcvzxx+fMM89MVeWEE07Ixo0bs3Xr1px//vm5/fbbU1XZsWPHI65x7bXXZvXq1bn44ouTJNu3b8+dd97pz9IAwDwjWM3i4IMPfuh4wYIFD7UXLFiQnTt3ZtWqVTn99NPziU98Ihs3bsyzn/3sR1yjtZarr746xxxzzP4qGwAYAY8C99LWrVuzePHiJMnll18+45izzjorl1xySR78g9fr16/fX+UBAPuRYLWXXve61+UNb3hDli9fnp07d844ZtWqVdmxY0dOPPHEHH/88Vm1atV+rhIA2B/qwVmUnzioamOS+5L8MMnO1tqKqnp8ko8kWZpkY5KXtNa+V1WV5F1JnpfkgSQXtNa+8pOuv2LFirZu3bqH9d16663WIO2Gfxtgr/Tevob5y9ZEM6qqm1prK2Y692hmrE5vrT112oVen+T61trRSa4f2klydpKjh5+VSd63Z2UD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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 사회, 경제적 지위별 차트\n",
        "bar_chart('Pclass') # 3st에 해당하는 사람들이 훨씬 더 많이 죽은 것을 확인 할 수 있다"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "id": "0majK_JWCK-b",
        "outputId": "98c14475-b713-4ce9-f0e1-259f09e8e85a"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 탑승지역(항구위치)별 차트\n",
        "bar_chart('Embarked') # Southampton에서 탑승한 사람들이 훨씬 더 많이 생존하고, 사망한 것을 알 수 있다"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "id": "bWinJWavC334",
        "outputId": "342d71ac-b1d2-4952-b189-8aa24ce8d3d4"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlYAAAFRCAYAAAC2SOM6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAWOklEQVR4nO3df7BnZX0f8PeHZWVNRI2wIYYlLuJiCsVfu3GoyaQI0YmmDcYmiuM0q8N0J1HS/LBjNck0bdNMSWYaStLWcRMzXTJJkGgp1NgmCsSQ+Ku7qKCxykYgLAVZNoISAgj99I970JXsZu/ufS7f7737es3c+Z7nOec853P/ufve85zzfKu7AwDA0h0z6wIAAFYLwQoAYBDBCgBgEMEKAGAQwQoAYBDBCgBgkGNnXUCSnHjiib1x48ZZlwEAcEi7du26p7vXH2jfXASrjRs3ZufOnbMuAwDgkKrqtoPtMxUIADCIYAUAMIhgBQAwyFw8YwUAHB2++tWvZs+ePXnwwQdnXcohrVu3Lhs2bMjatWsXfY5gBQA8Yfbs2ZPjjz8+GzduTFXNupyD6u7s27cve/bsyamnnrro80wFAgBPmAcffDAnnHDCXIeqJKmqnHDCCYd9Z02wAgCeUPMeqh5zJHUKVgDAUeeXfumXcuaZZ+Z5z3teXvCCF+RjH/vYkHE9YwUAzMzGt/3B0PFuvfgHDnnMRz7ykbzvfe/LDTfckOOOOy733HNPHn744SHXF6wAgKPKnXfemRNPPDHHHXdckuTEE08cNrapQADgqPLyl788t99+e04//fS86U1vyoc+9KFhY7tjBXCUOWvHWbMugRXipq03zbqEZfGUpzwlu3btyvXXX5/rrrsur33ta3PxxRfnDW94w5LHFqwAgKPOmjVrcs455+Scc87JWWedlR07dgwJVqYCAYCjyuc+97ncfPPNX2t/8pOfzLOe9awhY7tjBQAcVe6///78xE/8RO69994ce+yxec5znpPt27cPGVuwAgBmZjHLI4y2efPmfPjDH16WsU0FAgAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQBHnbvuuisXXHBBTjvttGzevDmvfOUr8/nPf37J41rHCgCYnX/9tMHj3XfIQ7o7P/RDP5StW7fm8ssvT5J86lOfyhe/+MWcfvrpS7q8YAUAHFWuu+66rF27Nj/2Yz/2tb7nP//5Q8Y2FQgAHFU+/elPZ/PmzcsytmAFADCIYAUAHFXOPPPM7Nq1a1nGFqwAgKPKueeem4ceeijbt2//Wt+NN96Y66+/fsljC1YAwFGlqnLllVfmgx/8YE477bSceeaZefvb355v+7ZvW/LYi3orsKpuTfKVJI8meaS7t1TVM5K8O8nGJLcmeU13f6mqKsmlSV6Z5IEkb+juG5ZcKQCw+ixieYTl8O3f/u254oorho97OHesXtrdL+juLVP7bUmu6e5NSa6Z2knyiiSbpp9tSd4xqlgAgHm2lKnA85PsmLZ3JHnVfv2X9YKPJnl6VT1zCdcBAFgRFrtAaCf5o6rqJO/s7u1JTuruO6f9dyU5ado+Ocnt+527Z+q7MwDM3E23/OWsS4BVa7HB6nu6+46q+tYkH6iq/7P/zu7uKXQtWlVty8JUYb7jO77jcE4FAJhLi5oK7O47ps+7k1yZ5MVJvvjYFN/0efd0+B1JTtnv9A1T3+PH3N7dW7p7y/r164/8NwAAmBOHDFZV9c1Vdfxj20lenuTTSa5OsnU6bGuSq6btq5P8aC04O8l9+00ZAgCsWou5Y3VSkj+tqk8l+XiSP+ju/5Xk4iQvq6qbk3zf1E6S9yf5QpLdSX4jyZuGVw0AsAR79uzJ+eefn02bNuXZz352Lrroojz00ENLHveQz1h19xeS/K2vfO7ufUnOO0B/J3nzkisDAFa9s3acNXS8m7bedMhjujuvfvWr8+M//uO56qqr8uijj2bbtm1561vfmksvvXRJ17fyOgBwVLn22muzbt26vPGNb0ySrFmzJpdcckkuu+yy3H///UsaW7ACAI4qn/nMZ7J58+Zv6HvqU5+ajRs3Zvfu3UsaW7ACABhEsAIAjipnnHFGdu3a9Q19X/7yl3PXXXfluc997pLGFqwAgKPKeeedlwceeCCXXXZZkuTRRx/NW97yllx00UV58pOfvKSxBSsA4KhSVbnyyivznve8J5s2bcoJJ5yQY445Jj/3cz+35LEX+5U2AADDLWZ5hOVwyimn5Oqrr06SfPjDH87rXve63HDDDXnRi160pHEFKwDgqPaSl7wkt91225CxTAUCAAwiWAEADCJYAQBPqIVvv5t/R1KnYAUAPGHWrVuXffv2zX246u7s27cv69atO6zzPLwOADxhNmzYkD179mTv3r2zLuWQ1q1blw0bNhzWOYIVAPCEWbt2bU499dRZl7FsTAUCAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAxy7KwLAOCJtfHB3511CawQt866gBXIHSsAgEEEKwCAQQQrAIBBBCsAgEEEKwCAQQQrAIBBBCsAgEEWHayqak1VfaKq3je1T62qj1XV7qp6d1U9aeo/bmrvnvZvXJ7SAQDmy+HcsfrJJJ/dr/3LSS7p7uck+VKSC6f+C5N8aeq/ZDoOAGDVW1SwqqoNSX4gyW9O7UpybpL3TIfsSPKqafv8qZ1p/3nT8QAAq9pi71j9xyRvTfL/pvYJSe7t7kem9p4kJ0/bJye5PUmm/fdNxwMArGqHDFZV9Y+S3N3du0ZeuKq2VdXOqtq5d+/ekUMDAMzEYu5YfXeSH6yqW5NcnoUpwEuTPL2qHvsS5w1J7pi270hySpJM+5+WZN/jB+3u7d29pbu3rF+/fkm/BADAPDhksOrut3f3hu7emOSCJNd29+uTXJfkh6fDtia5atq+empn2n9td/fQqgEA5tBS1rH6l0l+pqp2Z+EZqndN/e9KcsLU/zNJ3ra0EgEAVoZjD33I13X3Hyf542n7C0lefIBjHkzyIwNqAwBYUay8DgAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMMixsy6Agztrx1mzLoEV4qatN826BADijhUAwDCCFQDAIIIVAMAgghUAwCCCFQDAIIIVAMAgghUAwCCCFQDAIIIVAMAgghUAwCCCFQDAIIIVAMAgghUAwCDHzroADu6mW/5y1iUAAIfhkHesqmpdVX28qj5VVZ+pqn8z9Z9aVR+rqt1V9e6qetLUf9zU3j3t37i8vwIAwHxYzFTgQ0nO7e7nJ3lBku+vqrOT/HKSS7r7OUm+lOTC6fgLk3xp6r9kOg4AYNU7ZLDqBfdPzbXTTyc5N8l7pv4dSV41bZ8/tTPtP6+qaljFAABzalEPr1fVmqr6ZJK7k3wgyV8kube7H5kO2ZPk5Gn75CS3J8m0/74kJxxgzG1VtbOqdu7du3dpvwUAwBxYVLDq7ke7+wVJNiR5cZLvXOqFu3t7d2/p7i3r169f6nAAADN3WMstdPe9Sa5L8g+SPL2qHnurcEOSO6btO5KckiTT/qcl2TekWgCAObaYtwLXV9XTp+0nJ3lZks9mIWD98HTY1iRXTdtXT+1M+6/t7h5ZNADAPFrMOlbPTLKjqtZkIYhd0d3vq6o/T3J5Vf27JJ9I8q7p+Hcl+e2q2p3kr5JcsAx1AwDMnUMGq+6+MckLD9D/hSw8b/X4/geT/MiQ6gAAVhBfaQMAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMIhgBQAwiGAFADCIYAUAMMixsy6Ag9v44O/OugRWiFtnXQAASdyxAgAYRrACABhEsAIAGOSQwaqqTqmq66rqz6vqM1X1k1P/M6rqA1V18/T5LVN/VdWvVdXuqrqxql603L8EAMA8WMwdq0eSvKW7z0hydpI3V9UZSd6W5Jru3pTkmqmdJK9Ismn62ZbkHcOrBgCYQ4cMVt19Z3ffMG1/Jclnk5yc5PwkO6bDdiR51bR9fpLLesFHkzy9qp45vHIAgDlzWM9YVdXGJC9M8rEkJ3X3ndOuu5KcNG2fnOT2/U7bM/UBAKxqiw5WVfWUJO9N8lPd/eX993V3J+nDuXBVbauqnVW1c+/evYdzKgDAXFpUsKqqtVkIVb/T3f9t6v7iY1N80+fdU/8dSU7Z7/QNU9836O7t3b2lu7esX7/+SOsHAJgbi3krsJK8K8lnu/tX99t1dZKt0/bWJFft1/+j09uBZye5b78pQwCAVWsxX2nz3Un+aZKbquqTU9/PJrk4yRVVdWGS25K8Ztr3/iSvTLI7yQNJ3ji0YgCAOXXIYNXdf5qkDrL7vAMc30nevMS6AABWHCuvAwAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAwiWAEADCJYAQAMIlgBAAxyyGBVVb9VVXdX1af363tGVX2gqm6ePr9l6q+q+rWq2l1VN1bVi5azeACAebKYO1b/Ncn3P67vbUmu6e5NSa6Z2knyiiSbpp9tSd4xpkwAgPl3yGDV3X+S5K8e131+kh3T9o4kr9qv/7Je8NEkT6+qZ44qFgBgnh3pM1Yndfed0/ZdSU6atk9Ocvt+x+2Z+gAAVr0lP7ze3Z2kD/e8qtpWVTuraufevXuXWgYAwMwdabD64mNTfNPn3VP/HUlO2e+4DVPf39Ld27t7S3dvWb9+/RGWAQAwP440WF2dZOu0vTXJVfv1/+j0duDZSe7bb8oQAGBVO/ZQB1TV7yU5J8mJVbUnyS8kuTjJFVV1YZLbkrxmOvz9SV6ZZHeSB5K8cRlqBgCYS4cMVt39uoPsOu8Ax3aSNy+1KACAlcjK6wAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDCFYAAIMIVgAAgwhWAACDLEuwqqrvr6rPVdXuqnrbclwDAGDeDA9WVbUmyX9O8ookZyR5XVWdMfo6AADzZjnuWL04ye7u/kJ3P5zk8iTnL8N1AADmynIEq5OT3L5fe8/UBwCwqh07qwtX1bYk26bm/VX1uVnVwopzYpJ7Zl3EPKlfnnUFsCr42/I4/rYc1LMOtmM5gtUdSU7Zr71h6vsG3b09yfZluD6rXFXt7O4ts64DWF38bWGE5ZgK/N9JNlXVqVX1pCQXJLl6Ga4DADBXht+x6u5HquqiJH+YZE2S3+ruz4y+DgDAvFmWZ6y6+/1J3r8cY0NMIQPLw98Wlqy6e9Y1AACsCr7SBgBgEMEKAGAQwQoAYJCZLRAKALNSVT/zd+3v7l99omphdRGsmFtV9ZUkB327oruf+gSWA6wux0+fz03yXfn6eov/OMnHZ1IRq4K3Apl7VfWLSe5M8ttJKsnrkzyzu//VTAsDVryq+pMkP9DdX5naxyf5g+7+3tlWxkolWDH3qupT3f38Q/UBHK7pe2qf190PTe3jktzY3c+dbWWsVKYCWQn+uqpen+TyLEwNvi7JX8+2JGCVuCzJx6vqyqn9qiQ7ZlgPK5w7Vsy9qtqY5NIk352FYPVnSX6qu2+dXVXAalFVm5N8z9T8k+7+xCzrYWUTrAA46lXVtyZZ91i7u/9yhuWwglnHirlXVadX1TVV9emp/byq+vlZ1wWsfFX1g1V1c5Jbknxo+vyfs62KlUywYiX4jSRvT/LVJOnuG5NcMNOKgNXiF5OcneTz3X1qku9L8tHZlsRKJlixEnxTdz9+XZlHZlIJsNp8tbv3JTmmqo7p7uuSbJl1Uaxc3gpkJbinqk7LtFhoVf1wFta1Aliqe6vqKUmuT/I7VXV3vHXMEnh4nblXVc9Osj3JS5J8KQvPQLy+u2+baWHAildV35zkb7Iwg/P6JE9L8jvTXSw4bIIVc6+q1nT3o9MfwGMeWyEZYISqelaSTd39war6piRr/J3hSHnGipXglqranoUHTO+fdTHA6lFV/yzJe5K8c+o6Ocl/n11FrHSCFSvBdyb5YJI3ZyFk/aeq+p5DnAOwGG/OwuLDX06S7r45ybfOtCJWNMGKudfdD3T3Fd396iQvTPLULKw3A7BUD3X3w481qurYTC/KwJEQrFgRquofVtV/SbIrC6sjv2bGJQGrw4eq6meTPLmqXpbk95P8jxnXxArm4XXmXlXdmuQTSa5IcnV3exUaGKKqjklyYZKXJ6kkf5jkN9s/jhwhwYq5V1VP7e4vz7oOYHWqqvVJ0t17Z10LK59gxdyqqrd2969U1a/nAM88dPc/n0FZwCpQVZXkF5JclK8/FvNokl/v7n87s8JY8ay8zjz77PS5c6ZVAKvRT2fhbcDv6u5bkq8tRvyOqvrp7r5kptWxYrljxdyrqhd19w2zrgNYParqE0le1t33PK5/fZI/6u4XzqYyVjpvBbIS/Ieq+mxV/WJV/f1ZFwOsCmsfH6qSrz1ntXYG9bBKCFbMve5+aZKXJtmb5J1VdVNV/fyMywJWtoePcB/8nUwFsqJU1VlJ3prktd39pFnXA6xMVfVokgMt3VJJ1nW3u1YcEcGKuVdVfy/Ja5P8kyT7krw7yXu7++6ZFgYAjyNYMfeq6iNJLk/y+939f2ddDwAcjOUWmGtVtSbJLd196axrAYBD8fA6c627H01ySlV5ngqAueeOFSvBLUn+rKquzn4Pm3b3r86uJAD42wQrVoK/mH6OSXL8jGsBgIPy8DoAwCDuWDH3quq6HPhLmM+dQTkAcFCCFSvBv9hve10W1rN6ZEa1AMBBmQpkRaqqj3f3i2ddBwDszx0r5l5VPWO/5jFJtiR52ozKAYCDEqxYCXbl689YPZLk1iQXzqwaADgIwYq5VVXfleT27j51am/NwvNVtyb58xmWBgAHZOV15tk7kzycJFX1vUn+fZIdSe5Lsn2GdQHAAbljxTxb091/NW2/Nsn27n5vkvdW1SdnWBcAHJA7VsyzNVX1WPg/L8m1++3znwIA5o5/nJhnv5fkQ1V1T5K/SXJ9klTVc7IwHQgAc8U6Vsy1qjo7yTOT/FF3//XUd3qSp3T3DTMtDgAeR7ACABjEM1YAAIMIVgAAgwhWAACDCFYAAIMIVgAAg/x/03wMB0bDm8gAAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 타이타닉호에 탑승한 형제자매 수 별 차트\n",
        "bar_chart('SibSp') # 0명인 사람이 많이 죽기도 하였지만, 상대적으로 많이 생존한 것을 알 수 있다"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        },
        "id": "jcWFTAwADaKX",
        "outputId": "5f8ff93f-6f78-450c-e71a-4a4c68d0ca40"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ">>>##### 남성보단 여성이 많이 생존하였고, 1등급 승객, 가족이 있는 승객의 생존율이 더욱 높은 것을 알 수 있다. 탑승지역에서는 S 승객들이 많이 사망한 것으로 보인다"
      ],
      "metadata": {
        "id": "mdjRCRSqEJ2Y"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "9Y0te3t4Dy7q"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}