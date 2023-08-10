package com.strategyengine.xrpl.neuralnetwork;

import static com.google.common.base.Predicates.or;
import static springfox.documentation.builders.PathSelectors.regex;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.google.common.base.Predicate;

import springfox.documentation.builders.ApiInfoBuilder;
import springfox.documentation.builders.PathSelectors;
import springfox.documentation.builders.RequestHandlerSelectors;
import springfox.documentation.service.ApiInfo;
import springfox.documentation.spi.DocumentationType;
import springfox.documentation.spring.web.plugins.Docket;

/**
 * http://localhost:8060/swagger-ui.html
 * @author barry
 *
 */
@Configuration
public class SwaggerConfig {

	@Bean
	public Docket postsApi() {
		return new Docket(DocumentationType.SWAGGER_2).apiInfo(apiInfo())
		        .select()
		        .apis(RequestHandlerSelectors.basePackage("com.strategyengine.xrpl.neuralnetwork.rest.issuedtokens"))             
		          .paths(PathSelectors.any()).build();
	}

	private Predicate<String> postPaths() {
		return or(regex("/api/posts.*"), regex("/api/reports.*"));
	}

	private ApiInfo apiInfo() {
		return new ApiInfoBuilder().title("Find Global Id to XRPL Relationships")
				.description("API methods to help determine trustline holders via global id.")
				.license("Apache License v2.0")
				.version("1.0").build();
	}

}
