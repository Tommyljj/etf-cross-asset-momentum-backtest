"""Deterministic accounting tests. Synthetic fixtures are not market results."""
import unittest
import numpy as np
import pandas as pd
from src.backtest import simulate, build_schedule, momentum_scores
from src.metrics import performance_metrics, max_drawdown

class AccountingTests(unittest.TestCase):
    def market(self, opens, closes):
        idx = pd.bdate_range("2024-01-02",periods=len(opens))
        op = pd.DataFrame(opens,index=idx,columns=list("AB")[:len(opens[0])],dtype=float)
        cl = pd.DataFrame(closes,index=idx,columns=op.columns,dtype=float)
        return op,cl,pd.DataFrame(100.,index=idx,columns=op.columns)
    def sim(self, op, cl, vol, targets, cost=0):
        return simulate(op,cl,vol,{op.index[i]:pd.Series(w,index=op.columns,dtype=float)
                                   for i,w in targets.items()},op.index[0],cost,100.)
    def test_new_position_excludes_overnight_gap(self):
        op,cl,v=self.market([[100],[200]],[[100],[220]])
        r=self.sim(op,cl,v,{1:[1]})
        self.assertAlmostEqual(r.daily.nav.iloc[-1],110.)
    def test_old_overnight_and_new_intraday(self):
        op,cl,v=self.market([[100,100],[120,200]],[[100,100],[130,220]])
        r=self.sim(op,cl,v,{0:[1,0],1:[0,1]})
        self.assertAlmostEqual(r.daily.nav.iloc[-1],132.)
        self.assertAlmostEqual(r.holdings.asset_pnl.sum(),32.)
    def test_drift_without_daily_trading(self):
        op,cl,v=self.market([[100,100],[200,100],[100,100]],
                            [[100,100],[200,100],[100,100]])
        r=self.sim(op,cl,v,{0:[.5,.5]})
        self.assertAlmostEqual(r.daily.nav.iloc[-1],100.)
        self.assertEqual(len(r.trades),2)
        self.assertAlmostEqual(r.holdings.query("asset=='A'").weight.iloc[1],2/3)
    def test_same_names_rebalance_from_actual_weights(self):
        op,cl,v=self.market([[100,100],[200,100]],[[100,100],[200,100]])
        r=self.sim(op,cl,v,{0:[.5,.5],1:[.5,.5]},cost=.01)
        self.assertGreater(r.daily.fee_amount.iloc[1],0)
        self.assertTrue(np.allclose(r.holdings.iloc[-2:].weight,.5))
        self.assertGreaterEqual(r.daily.cash.min(),-1e-8)
    def test_entry_fee_self_financing(self):
        op,cl,v=self.market([[100]],[[100]])
        r=self.sim(op,cl,v,{0:[1]},cost=.01)
        self.assertAlmostEqual(r.daily.nav.iloc[0],100/1.01)
        self.assertAlmostEqual(r.daily.fee_amount.iloc[0],100-100/1.01)
    def test_cash_exit_cost(self):
        op,cl,v=self.market([[100],[100]],[[100],[100]])
        r=self.sim(op,cl,v,{0:[1],1:[0]},cost=.01)
        self.assertAlmostEqual(r.daily.nav.iloc[-1],100/1.01*.99)
        self.assertAlmostEqual(r.daily.cash.iloc[-1],r.daily.nav.iloc[-1])
        self.assertEqual(r.holdings.query("asset=='A'").adjusted_units.iloc[-1],0.0)
    def test_missing_open_skips_whole_order(self):
        op,cl,v=self.market([[100,100],[np.nan,100]],[[100,100],[np.nan,110]])
        r=self.sim(op,cl,v,{0:[1,0],1:[0,1]})
        self.assertEqual(r.executions.status.iloc[-1],"skipped_untradable")
        self.assertEqual(len(r.trades),1)
        self.assertAlmostEqual(r.daily.nav.iloc[-1],100.)
    def test_zero_volume_prevents_entry(self):
        op,cl,v=self.market([[100]],[[100]])
        v.iloc[0,0]=0
        r=self.sim(op,cl,v,{0:[1]})
        self.assertTrue(r.trades.empty)
        self.assertAlmostEqual(r.daily.cash.iloc[-1],100)
    def test_initial_loss_is_drawdown(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-.1,0.])), -.1)
    def test_standard_sharpe(self):
        r=pd.Series([.1,-.09,.02])
        m=performance_metrics(r)
        self.assertAlmostEqual(m["sharpe_ratio_rf0"],r.mean()/r.std(ddof=1)*np.sqrt(252))
        self.assertNotAlmostEqual(m["sharpe_ratio_rf0"],m["return_vol_ratio"])
    def test_schedule_and_all_negative_cash(self):
        dates=pd.bdate_range("2024-01-25","2024-02-05")
        close=pd.DataFrame({"A":np.arange(len(dates),0,-1.)+100},index=dates)
        targets,ranks=build_schedule(close,(1,),1,True)
        self.assertEqual(min(targets),pd.Timestamp("2024-02-01"))
        self.assertEqual(targets[min(targets)].sum(),0)
        self.assertEqual(len(ranks),2)
    def test_future_prices_do_not_change_prior_signals(self):
        dates=pd.bdate_range("2024-01-01","2024-03-04")
        close=pd.DataFrame({"A":100+np.arange(len(dates)), "B":200-np.arange(len(dates))},index=dates)
        before,_=build_schedule(close,(2,),1)
        close.loc["2024-02-02":,"B"]*=5
        after,_=build_schedule(close,(2,),1)
        pd.testing.assert_series_equal(before[pd.Timestamp("2024-02-01")],after[pd.Timestamp("2024-02-01")])
    def test_signal_rejects_gaps(self):
        close=pd.DataFrame({"A":[100,np.nan,110,120]})
        self.assertTrue(momentum_scores(close,(2,)).iloc[-1].isna().all())
    def test_empty_returns_rejected(self):
        with self.assertRaises(ValueError):
            performance_metrics(pd.Series(dtype=float))
    def test_daily_attribution_reconciles(self):
        op,cl,v=self.market([[100,100],[120,90],[130,80]],[[110,90],[125,85],[135,90]])
        r=self.sim(op,cl,v,{0:[.5,.5],2:[0,1]},.01)
        contributions=r.holdings.groupby("date").return_contribution.sum()
        np.testing.assert_allclose(contributions-r.daily.transaction_cost,r.daily.strategy_return,atol=1e-12)

if __name__=="__main__":
    unittest.main()
