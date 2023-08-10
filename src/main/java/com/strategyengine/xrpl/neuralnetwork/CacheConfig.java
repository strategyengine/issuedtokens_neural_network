package com.strategyengine.xrpl.neuralnetwork;

import java.util.concurrent.TimeUnit;

import org.springframework.cache.caffeine.CaffeineCache;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.github.benmanes.caffeine.cache.Caffeine;

//spring.cache.cache-names=reports-currency-cache
//spring.cache.caffeine.spec=maximumSize=500000,expireAfterAccess=1m

@Configuration
public class CacheConfig {

	public static final String CACHE_NAME_PREDICT = "predictCache";	
			


	@Bean(name = CACHE_NAME_PREDICT)
	public CaffeineCache tokenAmountReportCache() {
		return new CaffeineCache(CACHE_NAME_PREDICT,
				Caffeine.newBuilder().expireAfterWrite(1, TimeUnit.HOURS)
						.maximumSize(20000).recordStats().build());
	}

	
}
