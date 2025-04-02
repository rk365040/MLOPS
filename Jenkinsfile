pipeline {
    agent any

    environment {
        AWS_DEFAULT_REGION = 'us-east-1'
    }

    stages {
        stage('Clone Repository') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                script {
                    sh 'python3 -m venv venv'
                    sh 'source venv/bin/activate'
                    sh 'pip install boto3 nbformat sagemaker pandas numpy'
                }
            }
        }

        stage('Run SageMaker Notebook Script') {
            steps {
                script {
                    sh 'python3 create_and_run_notebook.py'
                }
            }
        }
    }
}
