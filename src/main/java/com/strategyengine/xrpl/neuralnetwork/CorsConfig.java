package com.strategyengine.xrpl.neuralnetwork;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {
	@Override
	public void addCorsMappings(CorsRegistry registry) {
	       registry.addMapping("/**").allowedOrigins("http://localhost:8090",
	    		   "http://local.strategyengine.one:8090",
	    		   "https://www.strategyengine.one",
	    		   "https://strategyengine.one",
	    		   "http://www.strategyengine.one",
	    		   "http://strategyengine.one").allowCredentials(true).allowedMethods("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS").maxAge(10);
	}
	
}
