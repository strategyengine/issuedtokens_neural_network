package com.strategyengine.xrpl.neuralnetwork.model;

import java.util.Date;

import com.fasterxml.jackson.annotation.JsonFormat;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;


@Builder
@EqualsAndHashCode
@ToString
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Setter
public class PredictionDate {

	@JsonFormat(pattern="yyyy-MM-dd")
	private Date date;
		
	private double price;

	
}
