-- Daily signups and orders for the last 14 days.
select date_trunc('day', created_at) as day,
       count(*) filter (where kind = 'signup') as signups,
       count(*) filter (where kind = 'order')  as orders
from events
where created_at >= now() - interval '14 days'
group by 1
order by 1;
