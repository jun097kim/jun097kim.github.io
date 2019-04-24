---
layout: post
title: React 앱 도커라이징과 CI/CD
---

Blockon 앱을 배포하면서 CI/CD의 험난한 여정을 소개한다.

서버사이드 렌더링용 서버를 분리하는 등 추후 확장성을 위해 docker-compose를 이용했다. API 서버로 돌아가는 **Express 컨테이너**와 static 파일 제공과 프록시를 담당하는 **Nginx 컨테이너** 2개가 동시에 돌아간다.

![CI/CD architecture](/assets/blockon-ci-cd-architecture.png){:width="500"}

원래는 Jenkins 도커 이미지를 이용하여 CI를 하려고 했으나 docker-compose 플러그인이 아직 완벽하지 않은 것 같아서 Travis CI를 사용했다.

개발용 서버와 배포용 서버 각각 1대씩 가동하고 있다고 가정한다. 우리 팀은 EC2를 사용하고 있기 때문에 CodeDeploy를 이용해 CD 환경을 구축했다.

## Travis CI 설정 파일 (.travis.yml)

<https://github.com/team-blockon/blockon/blob/master/.travis.yml>

내용이 많아 보이지만 크게 **빌드와 배포**로 나눌 수 있다. 먼저 빌드와 관련된 섹션들을 보겠다.

1. before_install 섹션: 최신 버전의 docker-compose를 설치한다. `docker-compose` 파일에 명시한 버전에 따라 호환되는 도커 버전 또한 다르다.

2. before_script 섹션: Travis에 등록한 환경변수를 `.env` 파일로 만들게 된다. 주의할 점은 `sh`로 쉘 스크립트를 실행하면 현재 쉘의 환경변수를 읽을 수 없게 된다. Travis가 넣어주는 환경변수를 이용하고자 한다면 반드시 `source` 명령어를 이용하자.

3. script 섹션: `docker-compose.yaml`에 정의된 내용을 바탕으로 빌드를 수행하고 잘 동작하는지 실제로 컨테이너를 실행해본다.

4. after_success 섹션: 도커 이미지를 배포를 위한 `.tar` 파일로 만든다.

### 이제 배포를 위한 준비가 완료되었으니 deploy 섹션을 살펴보자.

어떤 브랜치에 푸시하더라도 Travis는 빌드를 진행한다. 단, 다음 2개의 브랜치에 한해 배포까지 진행된다. **master 브랜치**에 푸시하는 경우 개발용 서버에 배포하고, **release 브랜치**에 푸시하는 경우 배포용 서버에 배포한다.

## Docker 설정 파일 (docker-compose.yaml)

<https://github.com/team-blockon/blockon/blob/master/docker-compose.yaml>

docker-compose 설정 파일은 Dockerfile을 기반으로 **이미지를 빌드**하거나, 만들어진 이미지 파일을 받아서 로드한 후 **컨테이너 실행**을 하기 위함이다.

Dockerfile과 각 컨테이너에 필요한 설정 파일은 compose 디렉토리에 서비스명으로 정리해두었다.

### 먼저 Express 이미지의 Dockerfile을 보자.

```dockerfile
FROM node:8-alpine
LABEL maintainer="jun097kim <jun097kim@gmail.com>"
RUN mkdir /app
RUN \
apk update &&\
apk add git
COPY . /app
WORKDIR /app/blockon-backend
RUN yarn
WORKDIR /app/blockon-frontend
RUN yarn
RUN yarn build
ADD compose/express/start.sh /start.sh
```

node:8-alpine을 베이스 이미지로 한다.

앱의 working directory로 사용할 `/app` 디렉토리를 만들어준 후, `yarn` 명령어를 이용하려면 git을 필요로 하므로 git을 설치한다.

그리고 현재 폴더(하위 폴더 포함)에 있는 모든 파일을 `/app` 경로에 복사한다.
디렉터리를 변경한 후 여러 명령어를 실행하고 싶을 때에는 `cd`가 아니라 `WORKDIR` 키워드를 이용하자.

backend, frontend 각각 yarn으로 모듈을 설치한다. frontend의 경우에는 `react build`까지 한다.

마지막으로 `ADD` 키워드를 이용해 로컬에 있는 `start.sh` 파일을 미리 컨테이너로 복사해 준다. 컨테이너가 실행되자 마자 실행될 파일이다.

### 다음은 Nginx 이미지의 Dockerfile을 보자.

```dockerfile
FROM nginx:1.15.5-alpine
LABEL maintainer="jun097kim <jun097kim@gmail.com>"

# 기본 설정파일을 새로운 파일로 대체

RUN rm -rf /etc/nginx/conf.d
COPY compose/nginx/conf.d /etc/nginx/conf.d
```

비교적 간단하다. `conf.d` 디렉터리를 로컬에 있는 것으로 통째로 교체한다.

## CodeDeploy 설정 파일 (appspec.yml)

### 먼저 files 섹션을 보자.

```yaml
files:
  - source: /
    destination: /home/ec2-user/blockon
```

CodeDeploy에서 각각의 배포 시도를 **개정**이라고 한다.

`source`는 개정에서 어떤 파일을 인스턴스로 복사할지를 나타낸다. 여기서는 모든 파일을 의미하는 /를 적어두었다.

### 다음은 hooks 섹션을 보자.

```yaml
hooks:
  Install:
    - location: scripts/load_image.sh
      runas: root
  ApplicationStart:
    - location: scripts/start_server.sh
      runas: root
```

CodeDeploy에는 **라이프사이클**이 있어서 특정 라이프사이클에 원하는 스크립트 실행이 가능하다.

여기서는 앱 시작 전에 필요한 도커 이미지를 가져오기 위해 `Install`, 앱 시작을 위해 `ApplicationStart` 라이프사이클을 이용한다.

### load_image 스크립트

```shell
docker rmi -f blockon_express
docker rmi -f blockon_nginx
docker load -i blockon_express.tar
docker load -i blockon_nginx.tar
```

Docker는 **이미지 태그**가 같으면 다시 로드하더라도 덮어쓰지 않는다. 그래서 원래 있던 이미지를 지우고, 다시 로드하는 것이다.

### start_server 스크립트

```shell
cd /opt/codedeploy-agent/deployment-root/$DEPLOYMENT_GROUP_ID/$DEPLOYMENT_ID/deployment-archive
docker rm -f blockon_express
docker rm -f blockon_nginx
/usr/local/bin/docker-compose -f docker-compose-deploy.yaml down
/usr/local/bin/docker-compose -f docker-compose-deploy.yaml up -d
```

컨테이너가 실행 중이면 컨테이너를 삭제한다. `docker-compose` 명령어로 방금 로드한 이미지로 컨테이너를 실행한다.

앞서 Travis CI, Docker, CodeDeploy 각각의 설정 파일들을 모두 살펴보았는데 내용이 많고 복잡할 수도 있다. 하지만 며칠 고생하더라도 한 번 적용해두면 프로젝트 전체 기간 동안 상당히 편해지는 것을 느낄 수 있을 것이다.
