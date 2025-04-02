pipeline {
    agent any

    environment {
        AWS_DEFAULT_REGION = 'us-west-1'
    }

    stages {
        stage('Clone Repository') {
            steps {
                script {
                    checkout([
                        $class: 'GitSCM',
                        branches: [[name: '*/main']],  // Replace 'main' with your branch name if different
                        userRemoteConfigs: [[
                            url: 'https://github.com/rk365040/MLOPS.git',
                            credentialsId: 'your-github-credentials-id'  // Add your Jenkins credentials ID for GitHub
                        ]]
                    ])
                }
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

        stage('Run SageMaker Script') {
            steps {
                script {
                    sh 'source venv/bin/activate && python3 create_and_run_notebook.py'
                }
            }
        }
    }
}
