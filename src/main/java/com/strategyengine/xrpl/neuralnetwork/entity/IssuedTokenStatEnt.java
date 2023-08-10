package com.strategyengine.xrpl.neuralnetwork.entity;

import java.math.BigDecimal;
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
@Table(name = "ISSUED_TOKEN_STAT")
@Builder(toBuilder = true)
@EqualsAndHashCode
@ToString
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Setter
public class IssuedTokenStatEnt {

	@Id
	@Column(name = "ID")
	@GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "issued_token_stat_generator")
	@SequenceGenerator(name="issued_token_stat_generator", sequenceName = "issued_token_stat_id_seq", allocationSize=1)
	private Integer id;

	@Column(name = "ISSUED_TOKEN_ID")
	private Integer issuedTokenId;
	
	
	@Column(name = "ISSUED_AMOUNT")
	private BigDecimal issuedAmount;
	
	@Column(name = "TRUSTLINES")
	private Integer trustlines;
	
	@Column(name = "CREATE_DATE")
	private Date createDate;
	
	@Column(name = "HOLDERS")
	private Integer holders;
	
	@Column(name = "OFFERS")
	private Integer offers;
	
	@Column(name = "PRICE")
	private BigDecimal price;
	
	@Column(name = "VOLUME_24H")
	private String volume24h;
	@Column(name = "VOLUME_7D")
	private String volume7d;
	@Column(name = "EXCHANGES_24H")
	private String exchanges24h;
	@Column(name = "EXCHANGES_7D")
	private String exchanges7d;
	
	
}
