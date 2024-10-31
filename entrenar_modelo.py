{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMRUZAMIatahYlVwu52aN1R",
      "include_colab_link": true
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
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ariatnajimenezg/Ariatna-Data-Sets/blob/main/entrenar_modelo.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9nmTkvE9GpnI",
        "outputId": "14df4925-6a28-4bc7-ec1e-221af9dbed68"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['modelo_clasificacion.pkl']"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ],
      "source": [
        "from sklearn.datasets import make_classification\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "import joblib\n",
        "\n",
        "# Genera un dataset de ejemplo\n",
        "X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)\n",
        "\n",
        "# Entrena el modelo\n",
        "model = RandomForestClassifier()\n",
        "model.fit(X, y)\n",
        "\n",
        "# Guarda el modelo entrenado\n",
        "joblib.dump(model, 'modelo_clasificacion.pkl')\n",
        "\n"
      ]
    }
  ]
}