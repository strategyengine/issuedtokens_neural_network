package com.strategyengine.xrpl.neuralnetwork.repo;

import org.springframework.data.jpa.repository.JpaRepository;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenEnt;


public interface IssuedTokenRepo extends JpaRepository<IssuedTokenEnt, Integer> {

}
