# --- build stage ---
FROM gradle:8.5-jdk17 AS build
WORKDIR /workspace

# Use the project's gradle wrapper for a consistent build.
COPY gradlew gradlew
COPY gradle gradle
COPY settings.gradle build.gradle ./
COPY src src

RUN sed -i 's/\r$//' gradlew \
 && chmod +x gradlew \
 && ./gradlew --no-daemon clean bootJar -x test

# --- runtime stage ---
FROM eclipse-temurin:17-jre-jammy
WORKDIR /app

# curl is required by the docker-compose healthcheck (replaces wget which the
# base image does not ship).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_OPTS="-Xms256m -Xmx512m -Dfile.encoding=UTF-8 -Duser.timezone=Asia/Seoul" \
    SPRING_PROFILES_ACTIVE=docker

COPY --from=build /workspace/build/libs/*.jar /app/app.jar

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar /app/app.jar"]
