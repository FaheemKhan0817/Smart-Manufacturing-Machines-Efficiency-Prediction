pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO = "faheemkhan08/gitops-project"
        DOCKER_HUB_CREDENTIALS_ID = 'gitops-dockerhub-token'
    }
    stages {
        stage('Checkout Github') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-tokens', url: 'https://github.com/FaheemKhan0817/Smart-Manufacturing-Machines-Efficiency-Prediction.git']])
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    echo 'Building Docker image...'
                    def dockerImage = docker.build("${DOCKER_HUB_REPO}:latest")
                    // Store the image in the script scope to avoid global variable issues
                    env.DOCKER_IMAGE = dockerImage.id
                }
            }
        }
        stage('Push Image to DockerHub') {
            steps {
                script {
                    echo 'Pushing Docker image to DockerHub...'
                    docker.withRegistry("https://registry.hub.docker.com", "${DOCKER_HUB_CREDENTIALS_ID}") {
                        def dockerImage = docker.image("${DOCKER_HUB_REPO}:latest")
                        dockerImage.push('latest')
                    }
                }
            }
        }
        stage('Install Kubectl & ArgoCD CLI') {
            steps {
                sh '''
                echo "Installing Kubectl & ArgoCD CLI..."
                # Install kubectl
                curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
                chmod +x kubectl
                mv kubectl /usr/local/bin/kubectl

                # Install ArgoCD CLI with a specific version and error handling
                ARGOCD_VERSION="v2.12.5"
                curl -sSL -o /usr/local/bin/argocd https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64
                if ! file /usr/local/bin/argocd | grep -q "ELF 64-bit"; then
                    echo "Error: Downloaded argocd binary is not an ELF executable. It might be an HTML error page."
                    cat /usr/local/bin/argocd
                    exit 1
                fi
                chmod +x /usr/local/bin/argocd
                '''
            }
        }
        stage('Apply Kubernetes & Sync App with ArgoCD') {
            steps {
                script {
                    kubeconfig(credentialsId: 'kubeconfig', serverUrl: 'https://192.168.49.2:8443') {
                        sh '''
                        argocd login 34.70.8.151:30128 --username admin --password $(kubectl get secret -n argocd argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d) --insecure
                        argocd app sync gitops-app
                        '''
                    }
                }
            }
        }
    }
}