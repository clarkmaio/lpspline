pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }

    stages {
        stage('Setup') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                // Install test dependencies if they are not in requirements.txt
                sh 'pip install pytest flake8' 
                sh 'pip install .'
            }
        }

        stage('Lint') {
            steps {
                sh 'flake8 .'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest'
            }
        }

        stage('Build') {
            steps {
                sh 'python setup.py sdist bdist_wheel'
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
