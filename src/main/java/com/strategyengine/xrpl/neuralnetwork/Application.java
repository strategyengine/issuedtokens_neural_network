package com.strategyengine.xrpl.neuralnetwork;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableScheduling;

import lombok.extern.log4j.Log4j2;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

@Log4j2
@EnableScheduling
@SpringBootApplication
@EnableCaching
@EnableSwagger2
public class Application {

	public static void main(String[] args) {
		SpringApplication.run(Application.class, args);
	}

}
