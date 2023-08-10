package com.strategyengine.xrpl.neuralnetwork.repo;

import java.util.Date;
import java.util.List;
import java.util.Map;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenStatEnt;


public interface IssuedTokenStatRepo extends JpaRepository<IssuedTokenStatEnt, Integer> {

	@Query(value="select max(create_date) from issued_token_stat", nativeQuery=true)
	Date getMaxDate();
	
	
	@Query(value="select icon, issued_token.id, diff/d.oldestAmt*100 prcnt, diff, issued_token_id, recentAmount, oldestAmt, currency, currency_hex, address, blackholed from"
			+ "			(select recent.issued_token_id, cast(recent.recentAmount as DECIMAL), cast(oldest.oldestAmt as DECIMAL), cast(oldest.oldestAmt as DECIMAL) - cast(recent.recentAmount AS DECIMAL) diff from"
			+ "			(select iti.issued_token_id, issued_amount recentAmount from ISSUED_TOKEN_STAT iti,"
			+ "			(select issued_token_id, max(create_date) recentCreateDt from ISSUED_TOKEN_STAT where create_date > :startDate group by issued_token_id) iss"
			+ "				where iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.recentCreateDt) as recent,"
			+ "			("
			+ "                select iti.issued_token_id, issued_amount oldestAmt, oldestCreateDt from ISSUED_TOKEN_STAT iti, ("
			+ "				select issued_token_id, min(create_date) oldestCreateDt from ISSUED_TOKEN_STAT where create_date > :startDate group by issued_token_id"
			+ "            ) as iss"
			+ "				where iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.oldestCreateDt"
			+ "            ) as oldest"
			+ "			where recent.issued_token_id = oldest.issued_token_id) d, issued_token"
			+ "			 where issued_token.id = d.issued_token_id and (diff > 0 or diff < 0) and blackholed = true"
			+ "			order by prcnt desc", nativeQuery=true)
	List<Map<String, Object>>  getCirculatingTokensByMostBurned(@Param("startDate") Date startDate);

	
	@Query(value="select icon, issued_token.id, diff/d.oldestAmt*100 prcnt, diff, issued_token_id, oldestAmt, recentAmount, currency, currency_hex, address, blackholed from"
			+ "			(select recent.issued_token_id, cast(recent.recentAmount as DECIMAL), cast(oldest.oldestAmt as DECIMAL),  cast(recent.recentAmount AS DECIMAL) - cast(oldest.oldestAmt as DECIMAL) diff from"
			+ "			(select iti.issued_token_id, holders recentAmount from ISSUED_TOKEN_STAT iti,"
			+ "			(select issued_token_id, max(create_date) recentCreateDt from ISSUED_TOKEN_STAT "
			+ "             where create_date > :startDate group by issued_token_id) iss"
			+ "				where iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.recentCreateDt"
			+ "            ) as recent,"
			+ "			("
			+ "                select iti.issued_token_id, holders oldestAmt, oldestCreateDt from ISSUED_TOKEN_STAT iti, ("
			+ "				select issued_token_id, min(create_date) oldestCreateDt from ISSUED_TOKEN_STAT where create_date > :startDate group by issued_token_id"
			+ "            ) as iss"
			+ "				where iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.oldestCreateDt"
			+ "            ) as oldest"
			+ "			where recent.issued_token_id = oldest.issued_token_id) d, issued_token"
			+ "			 where issued_token.id = d.issued_token_id and (diff > 0 or diff < 0) and blackholed = true"
			+ "			order by prcnt desc", nativeQuery=true)
	List<Map<String, Object>> getTokensHeldDesc(@Param("startDate") Date startDate);
	
	@Query(value="select icon, issued_token.id, diff/d.oldestAmt*100 prcnt, diff, issued_token_id, oldestAmt, recentAmount, currency, currency_hex, address, blackholed from"
			+ "			(select recent.issued_token_id, cast(recent.recentAmount as DECIMAL), cast(oldest.oldestAmt as DECIMAL),  cast(recent.recentAmount AS DECIMAL) - cast(oldest.oldestAmt as DECIMAL) diff from"
			+ "			(select iti.issued_token_id, holders recentAmount from ISSUED_TOKEN_STAT iti,"
			+ "			(select issued_token_id, max(create_date) recentCreateDt from ISSUED_TOKEN_STAT "
			+ "             where create_date > :startDate group by issued_token_id) iss"
			+ "				where iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.recentCreateDt"
			+ "            ) as recent,"
			+ "			("
			+ "                select iti.issued_token_id, holders oldestAmt, oldestCreateDt from ISSUED_TOKEN_STAT iti, ("
			+ "				select issued_token_id, min(create_date) oldestCreateDt from ISSUED_TOKEN_STAT where create_date > :startDate group by issued_token_id"
			+ "            ) as iss"
			+ "				where iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.oldestCreateDt"
			+ "            ) as oldest"
			+ "			where recent.issued_token_id = oldest.issued_token_id) d, issued_token"
			+ "			 where issued_token.id = d.issued_token_id and (diff > 0 or diff < 0) and blackholed = true"
			+ "			order by prcnt asc", nativeQuery=true)
	List<Map<String, Object>> getTokensHeldAsc(@Param("startDate") Date startDate);
	
	@Query(value="select icon, issued_token.id, diff/d.oldestAmt*100 prcnt, diff, issued_token_id, oldestAmt, recentAmount, currency, currency_hex, address, blackholed from"
			+ "			(select recent.issued_token_id, cast(recent.recentAmount as DECIMAL), cast(oldest.oldestAmt as DECIMAL),  cast(recent.recentAmount AS DECIMAL) - cast(oldest.oldestAmt as DECIMAL) diff from"
			+ "			(select iti.issued_token_id, (cast(issued_amount as DECIMAL)*cast(price as DECIMAL))/(holders*offers) recentAmount from ISSUED_TOKEN_STAT iti,"
			+ "			(select issued_token_id, max(create_date) recentCreateDt from ISSUED_TOKEN_STAT "
			+ "             where create_date > :startDate group by issued_token_id) iss"
			+ "				where holders>0 and offers>0 and iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.recentCreateDt"
			+ "            ) as recent,"
			+ "			("
			+ "                select iti.issued_token_id, (cast(issued_amount as DECIMAL)*cast(price as DECIMAL))/(holders*offers) oldestAmt, oldestCreateDt from ISSUED_TOKEN_STAT iti, ("
			+ "				select issued_token_id, min(create_date) oldestCreateDt from ISSUED_TOKEN_STAT where create_date > :startDate group by issued_token_id"
			+ "            ) as iss"
			+ "				where holders>0 and offers>0 and iss.issued_token_id = iti.issued_token_id and iti.create_date = iss.oldestCreateDt"
			+ "            ) as oldest"
			+ "			where recent.issued_token_id = oldest.issued_token_id) d, issued_token"
			+ "			 where d.oldestAmt > 0 and issued_token.id = d.issued_token_id and (diff > 0 or diff < 0) and blackholed = true"
			+ "			order by prcnt asc", nativeQuery=true)
	List<Map<String, Object>> getTokensValueByHeldOffersAsc(@Param("startDate") Date startDate);
	
	
	
}
