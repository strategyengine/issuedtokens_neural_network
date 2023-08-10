package com.strategyengine.xrpl.neuralnetwork.entity;

import java.util.Date;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.SequenceGenerator;
import javax.persistence.Table;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Entity
@Table(name = "ISSUED_TOKEN")
@Builder(toBuilder = true)
@EqualsAndHashCode
@ToString
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Setter
public class IssuedTokenEnt {

	@Id
	@Column(name = "ID")
	@GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "issued_token_generator")
	@SequenceGenerator(name="issued_token_generator", sequenceName = "issued_token_id_seq", allocationSize=1)
	private Integer id;

	@Column(name = "ADDRESS")
	private String xrpAddress;
	
	@Column(name = "CREATE_DATE")
	private Date createDate;
	
	@Column(name = "UPDATE_DATE")
	private Date updateDate;
	
	@Column(name = "CURRENCY")
	private String currency;

	@Column(name = "CURRENCY_HEX")
	private String currencyHex;
	
	@Column(name = "INTERNET_DOMAIN")
	private String domain;
	
	@Column(name = "VERIFIED")
	private Boolean verified;
	
	@Column(name = "USERNAME")
	private String username;
	
	@Column(name = "TWITTER")
	private String twitter;
	
	@Column(name = "KYC")
	private Boolean kyc;
	
	@Column(name = "BLACKHOLED")
	private Boolean blackholed;
	
	@Column(name = "ICON")
	private String icon;
	
	
	
}
