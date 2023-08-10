# For Java 11, try this
#FROM adoptopenjdk/openjdk11:alpine-jre
FROM adoptopenjdk/openjdk11:jre-11.0.11_9

#RUN apt-get update; apt-get install -y fontconfig libfreetype6 ; apt-get install -y libfontconfig1

# Refer to Maven build -> finalName
ARG JAR_FILE=target/issuedtoken_neural_network-0.0.1-SNAPSHOT.jar

# cd /opt/app
WORKDIR /opt/app

# cp target/issuedtoken_neural_network-0.0.1-SNAPSHOT.jar /opt/app/app.jar
COPY ${JAR_FILE} app.jar

EXPOSE 8080

# java -jar /opt/app/app.jar
ENTRYPOINT ["java","-jar","app.jar"]